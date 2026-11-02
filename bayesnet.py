"""
Classificador de Rede Bayesiana (BayesNet) discreta, implementado com numpy.

Estruturas suportadas:
  - "naive": Naive Bayes (a classe é pai de todos os atributos).
  - "tan"  : Tree-Augmented Naive Bayes. Além da classe, cada atributo pode ter
             UM outro atributo como pai; a árvore é aprendida pela informação
             mútua condicional (algoritmo de Chow-Liu / árvore geradora máxima).
             É a mesma ideia do BayesNet do Weka com busca TAN.

Atributos contínuos são discretizados por quantis DENTRO do fit(), então, ao
usar validação cruzada, nenhuma informação do fold de teste vaza para o treino.
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class BayesNetClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_bins=5, estrutura="tan", alpha=1.0):
        self.n_bins = n_bins          # nº de faixas por atributo
        self.estrutura = estrutura    # "naive" ou "tan"
        self.alpha = alpha            # suavização de Laplace

    # ------------------------------------------------------------------ #
    def _discretizar(self, X):
        Xd = np.empty(X.shape, dtype=int)
        for j, bordas in enumerate(self.bordas_):
            Xd[:, j] = np.searchsorted(bordas, X[:, j], side="right")
        return Xd

    def _aprender_tan(self, Xd, yi, K):
        """Pesos = I(Xi; Xj | C); depois árvore geradora máxima (Prim)."""
        d = Xd.shape[1]
        a = self.alpha
        W = np.zeros((d, d))
        for i in range(d):
            for j in range(i + 1, d):
                Si, Sj = self.n_estados_[i], self.n_estados_[j]
                cont = np.zeros((K, Si, Sj))
                np.add.at(cont, (yi, Xd[:, i], Xd[:, j]), 1)
                cont += a / (Si * Sj)
                p_abc = cont / cont.sum()
                p_c = p_abc.sum(axis=(1, 2), keepdims=True)
                p_ab_c = p_abc / p_c
                p_a_c = p_ab_c.sum(axis=2, keepdims=True)
                p_b_c = p_ab_c.sum(axis=1, keepdims=True)
                W[i, j] = W[j, i] = np.sum(p_abc * np.log(p_ab_c / (p_a_c * p_b_c)))

        pais = [-1] * d                 # -1 => só a classe é pai
        na_arvore, fora = [0], set(range(1, d))
        while fora:
            melhor, peso = None, -np.inf
            for i in na_arvore:
                for j in fora:
                    if W[i, j] > peso:
                        peso, melhor = W[i, j], (i, j)
            i, j = melhor
            pais[j] = i                 # aresta orientada i -> j
            na_arvore.append(j)
            fora.remove(j)
        return pais

    # ------------------------------------------------------------------ #
    def fit(self, X, y):
        if self.estrutura not in ("naive", "tan"):
            raise ValueError("estrutura deve ser 'naive' ou 'tan'")
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        n, d = X.shape
        self.n_features_in_ = d
        self.classes_, yi = np.unique(y, return_inverse=True)
        K = len(self.classes_)

        quantis = np.linspace(0, 1, self.n_bins + 1)[1:-1]
        self.bordas_ = [np.unique(np.quantile(X[:, j], quantis)) for j in range(d)]
        self.n_estados_ = [len(b) + 1 for b in self.bordas_]
        Xd = self._discretizar(X)

        self.pais_ = [-1] * d
        if self.estrutura == "tan" and d > 1:
            self.pais_ = self._aprender_tan(Xd, yi, K)

        a = self.alpha
        cont_c = np.bincount(yi, minlength=K)
        self.log_prior_ = np.log((cont_c + a) / (n + a * K))

        self.log_cpt_ = []
        for j in range(d):
            S, p = self.n_estados_[j], self.pais_[j]
            if p == -1:
                cont = np.zeros((K, S))
                np.add.at(cont, (yi, Xd[:, j]), 1)
            else:
                cont = np.zeros((K, self.n_estados_[p], S))
                np.add.at(cont, (yi, Xd[:, p], Xd[:, j]), 1)
            cont += a
            cont /= cont.sum(axis=-1, keepdims=True)
            self.log_cpt_.append(np.log(cont))
        return self

    # ------------------------------------------------------------------ #
    def _log_conjunta(self, X):
        X = np.asarray(X, dtype=float)
        Xd = self._discretizar(X)
        L = np.tile(self.log_prior_, (len(Xd), 1))
        for j, tab in enumerate(self.log_cpt_):
            p = self.pais_[j]
            if p == -1:
                L += tab[:, Xd[:, j]].T
            else:
                L += tab[:, Xd[:, p], Xd[:, j]].T
        return L

    def predict_proba(self, X):
        L = self._log_conjunta(X)
        L = L - L.max(axis=1, keepdims=True)
        P = np.exp(L)
        return P / P.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[np.argmax(self._log_conjunta(X), axis=1)]

    def arestas(self, nomes):
        """Lista legível da estrutura aprendida (pai -> filho)."""
        ar = [("Classe", nomes[j]) for j in range(len(nomes))]
        ar += [(nomes[p], nomes[j]) for j, p in enumerate(self.pais_) if p != -1]
        return ar
