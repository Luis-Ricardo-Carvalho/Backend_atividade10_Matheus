import json
import os
import pickle

import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS

AQUI = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
CORS(app)  # permite requisições de origens diferentes (o front React)

# Carrega os arquivos gerados por modelo.py
model = pickle.load(open(os.path.join(AQUI, "model.pkl"), "rb"))
names = pickle.load(open(os.path.join(AQUI, "names.pkl"), "rb"))
info = json.load(open(os.path.join(AQUI, "info.json"), encoding="utf-8"))
FEATURES = info["features"]


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "API Online",
        "message": "API de Machine Learning (classificador de rochas ígneas - BayesNet) em execução!",
        "versao": "1.0",
    })


@app.route("/api/info", methods=["GET"])
def get_info():
    """Atributos esperados, classes, faixas de treino e métricas (usado pelo front)."""
    return jsonify({
        "features": FEATURES,
        "classes": names,
        "faixas": info["faixas"],
        "teste": {"acuracia": info["teste"]["acuracia"], "f1_macro": info["teste"]["f1_macro"]},
        "validacao_cruzada": info["validacao_cruzada"],
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    dados = request.get_json(force=True, silent=True) or {}
    # Exemplo: {"SiO2": 50, "TiO2": 1.3, "Al2O3": 17, "FeO": 9.5, "MnO": 0.17,
    #           "MgO": 8, "CaO": 10.5, "Na2O": 2.8, "K2O": 0.5, "P2O5": 0.25}
    faltando = [f for f in FEATURES if f not in dados]
    if faltando:
        return jsonify({"erro": f"Campos ausentes: {faltando}", "esperado": FEATURES}), 400
    try:
        valores = [float(dados[f]) for f in FEATURES]
    except (TypeError, ValueError):
        return jsonify({"erro": "Todos os campos devem ser numéricos."}), 400
    if not np.all(np.isfinite(valores)):
        return jsonify({"erro": "Valores inválidos (NaN/infinito)."}), 400

    proba = model.predict_proba(np.array([valores]))[0]
    idx = int(np.argmax(proba))
    ordem = np.argsort(-proba)
    resposta = {
        "codigo": idx,
        "predicao": names[idx],
        "confianca": round(float(proba[idx]), 4),
        "probabilidades": {names[i]: round(float(proba[i]), 4) for i in ordem},
        "dados_recebidos": dados,
    }
    soma = sum(valores)
    if abs(soma - 100) > 2:
        resposta["aviso"] = (f"A soma dos óxidos é {soma:.1f}%. O modelo foi treinado com "
                             "dados normalizados a 100%.")
    return jsonify(resposta)


if __name__ == "__main__":
    app.run(port=int(os.environ.get("PORT", 5001)))
