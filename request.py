import requests

url = 'http://127.0.0.1:5001/api/predict'
dados_amostra = {
    "SiO2": 50.0, "TiO2": 1.3, "Al2O3": 17.0, "FeO": 9.5, "MnO": 0.17,
    "MgO": 8.0, "CaO": 10.5, "Na2O": 2.8, "K2O": 0.5, "P2O5": 0.25
}

# Realiza a requisição POST enviando os dados em JSON
resposta = requests.post(url, json=dados_amostra)

print("Código HTTP de Status:", resposta.status_code)
print("Resposta JSON da API:", resposta.json())
