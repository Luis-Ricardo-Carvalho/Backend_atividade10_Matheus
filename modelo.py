import datetime as dt
import json
import os
import pickle
import re
import sys

import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_validate, train_test_split)

from bayesnet import BayesNetClassifier

CSV_URL = ("https://docs.google.com/spreadsheets/d/"
           "1vgYehDFCOsKIqVkCBy2JbizU-QzLfhQsKekmf9lRVC8/export?format=csv")
MIN_AMOSTRAS = 20     # classes com menos exemplos que isso são descartadas
SEMENTE = 50
# Rochas plutônicas têm a mesma química das vulcânicas; só a textura muda (e a química não vê textura).
# Conferido nos dados: SiO2 médio gabro 51.7 ~ basalto 50.6; diorito 60.5 ~ andesito 59.5;
# granodiorito 65.6 ~ dacito 66.8; granito 72.2 ~ riolito 74.6.
EQUIVALENTES = {"Gabbro": "Basalt", "Diorite": "Andesite", "Quartz diorite": "Andesite",
                "Granodiorite": "Dacite", "Granite": "Rhyolite"}

# ----------------------------------------------------------------------------
# 1. Carregar os dados
# ----------------------------------------------------------------------------
AGRUPAR = "--sem-agrupar" not in sys.argv   # padrão: agrupa; use --sem-agrupar para manter as 10 classes
args = [a for a in sys.argv[1:] if not a.startswith("--")]
if args:
    origem = args[0]
elif os.path.exists("Rocks_Database.xlsx"):
    origem = "Rocks_Database.xlsx"
else:
    origem = CSV_URL
df = pd.read_excel(origem) if origem.lower().endswith((".xlsx", ".xls")) else pd.read_csv(origem)
print(f"Dados lidos de: {origem}  ->  {df.shape[0]} linhas, {df.shape[1]} colunas")

# ----------------------------------------------------------------------------
# 2. Pré-processamento
# ----------------------------------------------------------------------------
def limpar_coluna(c):
    c = c.strip()
    return c if c == "rock_name" else c.replace("*", "").rstrip("n")  # SiO2n->SiO2, FeO*n->FeO

df.columns = [limpar_coluna(c) for c in df.columns]
FEATURES = [c for c in df.columns if c != "rock_name"]

TERMOS = ["dike", "sill", "intrusion", "tuff", "breccia", "vitrophyre", "plug",
          "scoria", "hydroclastic", "volcanic"]

def normalizar_nome(nome):
    """'Basaltic andesite dike' -> 'Basaltic andesite'; nomes mistos/ambíguos -> None."""
    n = str(nome).lower().replace("?", "").strip()
    if "/" in n or " and " in n:
        return None
    for t in TERMOS:
        n = re.sub(rf"\b{t}\b", "", n)
    n = re.sub(r"\s+", " ", n).strip()
    n = (n.replace("micro-quartz diorite", "quartz diorite")
          .replace("microdiorite", "diorite").replace("microgabbro", "gabbro"))
    return n.capitalize() if n and n != "nan" else None

n0 = len(df)
datas_recuperadas = 0

def para_numero(v):
    """O Excel transformou valores como 1.12 em datas (1 de dezembro). Recupera: dia + mês/100."""
    global datas_recuperadas
    if isinstance(v, (dt.datetime, dt.date, pd.Timestamp)):
        datas_recuperadas += 1
        return v.day + v.month / 100
    return pd.to_numeric(v, errors="coerce")                 # '03.09' -> 3.09 ; '<0.05' -> NaN

df = df.dropna(how="all")                                    # linhas em branco
for c in FEATURES:
    df[c] = df[c].map(para_numero)
df = df.dropna(subset=["rock_name"])
df = df[df[FEATURES].isna().sum(axis=1) <= 2]                # MnO/P2O5 ausentes são comuns; mais que isso, descarta
soma = df[FEATURES].sum(axis=1)                              # NaN conta como 0
df = df[(soma >= 98) & (soma <= 102)]                        # dados normalizados a ~100%
df["classe"] = df["rock_name"].map(normalizar_nome)
df = df.dropna(subset=["classe"])
if AGRUPAR:
    df["classe"] = df["classe"].replace(EQUIVALENTES)
df = df.drop_duplicates(subset=FEATURES + ["classe"])        # evita repetição treino/teste
contagem = df["classe"].value_counts()
mantidas = contagem[contagem >= MIN_AMOSTRAS].index
df = df[df["classe"].isin(mantidas)].reset_index(drop=True)
faltantes = df[FEATURES].isna().sum()
print(f"Valores em formato de data recuperados: {datas_recuperadas}")
print("Valores ausentes (serão imputados pela mediana dentro de cada fold):")
print(faltantes[faltantes > 0].to_string() if faltantes.sum() else "  nenhum")
print(f"Após limpeza: {len(df)} de {n0} linhas | classes mantidas:")
print(df["classe"].value_counts().to_string(), "\n")

X, y = df[FEATURES].values, df["classe"].values

# ----------------------------------------------------------------------------
# 3. Divisão treino/teste (estratificada) + validação cruzada
# ----------------------------------------------------------------------------
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEMENTE)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMENTE)

def com_imputacao(est):
    """Imputa ausentes pela mediana do treino (dentro de cada fold) e depois treina o modelo."""
    return Pipeline([("imp", SimpleImputer(strategy="median")), ("modelo", est)])

candidatos = {
    "Árvore de decisão (base)": com_imputacao(tree.DecisionTreeClassifier(random_state=SEMENTE)),
    "Naive Bayes": com_imputacao(BayesNetClassifier(estrutura="naive")),
    "BayesNet (TAN)": com_imputacao(BayesNetClassifier(estrutura="tan")),
}
resultados_cv = {}
print("Validação cruzada (5-fold) no conjunto de treino:")
for nome, m in candidatos.items():
    r = cross_validate(m, X_tr, y_tr, cv=cv,
                       scoring={"acc": "accuracy", "f1": "f1_macro"})
    resultados_cv[nome] = {
        "acuracia_media": float(r["test_acc"].mean()), "acuracia_dp": float(r["test_acc"].std()),
        "f1_macro_medio": float(r["test_f1"].mean()), "f1_macro_dp": float(r["test_f1"].std())}
    print(f"  {nome:26s} acc={r['test_acc'].mean():.3f}±{r['test_acc'].std():.3f}  "
          f"F1-macro={r['test_f1'].mean():.3f}±{r['test_f1'].std():.3f}")

# ----------------------------------------------------------------------------
# 4. Treinar a BayesNet (ajuste de n_bins e estrutura por validação cruzada)
# ----------------------------------------------------------------------------
grade = GridSearchCV(com_imputacao(BayesNetClassifier()),
                     {"modelo__n_bins": [3, 4, 5, 6, 8], "modelo__estrutura": ["naive", "tan"]},
                     cv=cv, scoring="f1_macro")
grade.fit(X_tr, y_tr)
melhor = grade.best_estimator_
print(f"\nMelhores hiperparâmetros: {grade.best_params_} (F1-macro CV={grade.best_score_:.3f})")

# ----------------------------------------------------------------------------
# 5. Avaliação no conjunto de teste
# ----------------------------------------------------------------------------
pred = melhor.predict(X_te)
acc = accuracy_score(y_te, pred)
f1m = f1_score(y_te, pred, average="macro")
classes = list(melhor.classes_)
cm = confusion_matrix(y_te, pred, labels=classes)
relatorio = classification_report(y_te, pred, output_dict=True, zero_division=0)
print(f"\nTESTE: acurácia={acc:.3f}  F1-macro={f1m:.3f}\n")
print(classification_report(y_te, pred, zero_division=0))
print("Matriz de confusão (linhas = real, colunas = previsto):")
print(pd.DataFrame(cm, index=classes, columns=classes).to_string())

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Previsto"); ax.set_ylabel("Real"); ax.set_title("Matriz de confusão (teste)")
    fig.tight_layout(); fig.savefig("matriz_confusao.png", dpi=150)
except ImportError:
    print("(matplotlib não instalado: imagem da matriz de confusão não gerada)")

# ----------------------------------------------------------------------------
# 6. Reajustar com TODOS os dados e exportar (como no projeto base)
# ----------------------------------------------------------------------------
melhores = {k.replace("modelo__", ""): v for k, v in grade.best_params_.items()}
final = com_imputacao(BayesNetClassifier(**melhores)).fit(X, y)
pickle.dump(final, open("model.pkl", "wb"))            # modelo
pickle.dump(list(final.classes_), open("names.pkl", "wb"))  # nomes das classes

info = {
    "features": FEATURES,
    "classes": list(final.classes_),
    "faixas": {f: [float(df[f].min()), float(df[f].max())] for f in FEATURES},
    "limpeza": {"linhas_originais": int(n0), "linhas_usadas": int(len(df)),
                "datas_recuperadas": int(datas_recuperadas),
                "valores_imputados": {k: int(v) for k, v in faltantes.items() if v > 0}},
    "amostras_por_classe": {k: int(v) for k, v in df["classe"].value_counts().items()},
    "hiperparametros": melhores,
    "estrutura_aprendida": [list(a) for a in final.named_steps["modelo"].arestas(FEATURES)],
    "validacao_cruzada": resultados_cv,
    "teste": {"acuracia": acc, "f1_macro": f1m, "classes": classes,
              "matriz_confusao": cm.tolist(), "relatorio": relatorio},
}
json.dump(info, open("info.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("\nModelo, nomes e info.json exportados com sucesso!")
