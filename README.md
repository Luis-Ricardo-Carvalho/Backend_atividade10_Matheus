# Classificador de Rochas Ígneas com Rede Bayesiana (BayesNet)

Dados (rochas_corrigido.csv) → pré-processamento → treino e validação cruzada → API Flask → aplicação React.

## Como rodar

```bash
# 1) Backend
python -m venv venv && venv\Scripts\activate        # Windows
pip install -r requirements.txt
python modelo.py        # (opcional) retreina e regenera model.pkl, names.pkl, info.json
python app.py           # API em http://127.0.0.1:5001
python request.py       # (opcional) testa a API

# 2) Frontend (outro terminal)
cd frontend
npm install
npm run dev             # http://localhost:5173
```

`model.pkl`, `names.pkl` e `info.json` já vêm treinados com `rochas_corrigido.csv`
(scikit-learn 1.8.0, numpy 2.4.4, Python 3.12). Se der erro ao carregar o modelo por diferença
de versões, rode `python modelo.py` para regenerá-los no seu ambiente.

Se a API estiver publicada (ex.: Azure), copie `frontend/.env.example` para `frontend/.env`
e troque `VITE_API_URL`. Faça commit dos três arquivos gerados (a API precisa deles no deploy).

## Arquivos

| Arquivo | Função |
|---|---|
| `bayesnet.py` | Classificador de Rede Bayesiana (Naive Bayes ou TAN), com discretização e suavização de Laplace |
| `modelo.py` | Limpeza, divisão treino/teste, validação cruzada 5-fold, ajuste, avaliação e exportação |
| `app.py` | API Flask: `GET /`, `GET /api/info`, `POST /api/predict` |
| `request.py` | Exemplo de consumo da API |
| `frontend/` | Aplicação React (Vite) com formulário dos 10 óxidos |
| `rochas_corrigido.csv` / `Rocks_Database_corrigido.xlsx` | Dados corrigidos (datas recuperadas como números, linhas em branco removidas) |
| `matriz_confusao.png` | Matriz de confusão no conjunto de teste |

## Pré-processamento (modelo.py)

- **Datas no lugar de números:** o Excel converteu 3.400 valores em datas (ex.: `1.12` virou 1/dez).
  Cada um foi recuperado como dia + mês/100 e já está corrigido em `rochas_corrigido.csv`
  (o `modelo.py` também sabe corrigir a planilha original). Conferência: depois disso a soma dos óxidos
  fica em 100,00 ± 0,05 em praticamente todas as linhas.
- Remove linhas em branco, sem nome de rocha ou com mais de 2 valores ausentes; mantém soma dos óxidos entre 98 e 102.
- **Valores ausentes** (MnO: 149, P2O5: 208, Na2O: 1) são imputados pela mediana, dentro de cada fold.
- Padroniza `rock_name` (`Andesite dike`, `Andesite sill`… → `Andesite`), descarta nomes ambíguos
  (`Intrusion`, `X/Y sill`) e classes com menos de 20 amostras.
- **Agrupamento por equivalência química** (padrão): Gabbro→Basalt, Diorite e Quartz diorite→Andesite,
  Granodiorite→Dacite, Granite→Rhyolite. Rochas plutônicas e vulcânicas têm a mesma composição
  (o SiO2 médio é equivalente nos dados) e diferem só na textura, que a química não captura.
  Para manter as 10 classes originais: `python modelo.py --sem-agrupar`.
- Remove linhas duplicadas; discretiza os óxidos em faixas por quantis **dentro de cada fold**
  (sem vazamento de dados).

## Resultados (4.008 linhas usadas, 5 classes, 20% para teste)

| Modelo | Acurácia (CV 5-fold) | F1-macro (CV) |
|---|---|---|
| Naive Bayes | 0,773 ± 0,006 | 0,769 ± 0,009 |
| **BayesNet (TAN, 8 faixas)** | 0,794 ± 0,011 | 0,790 ± 0,017 |
| Árvore de decisão (modelo do projeto base) | 0,824 ± 0,015 | 0,816 ± 0,018 |

**Teste (BayesNet TAN):** acurácia 0,843 · F1-macro 0,832. Sem agrupar (10 classes): 0,779 · 0,551.

- A TAN supera o Naive Bayes (ganho de cerca de 2 pontos), porque modela dependências entre óxidos
  (ex.: CaO→MgO, SiO2→FeO).
- Os erros ocorrem principalmente entre classes vizinhas na escala de sílica
  (basalto ↔ andesito basáltico, andesito ↔ dacito), onde as faixas se sobrepõem.
- A árvore de decisão teve acurácia ~3 pontos maior na validação cruzada; a BayesNet, por outro lado, devolve
  probabilidades por classe, que o front exibe.
