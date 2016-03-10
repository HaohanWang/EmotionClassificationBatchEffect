__author__ = 'Haohan Wang'

import scipy
import scipy.linalg as linalg
import scipy.optimize as opt
import time
import numpy as np

from sklearn.svm import SVR
from sklearn.naive_bayes import GaussianNB

class LMM:
    def __init__(self, basicClassifier='svm'):
        if basicClassifier == 'svm':
            self.clf = SVR(tol=1e-5, C=5)
        else:
            self.clf = GaussianNB()
        self.K = None
        self.K = None
        self.numintervals = 100
        self.ldeltamin = -5
        self.ldeltamax = 5
        self.scale = 0
        self.KSquare = False
        self.KSearch = False

    def setK(self, K):
        self.K = K

    def setKv(self, Kv):
        self.Kv = Kv

    def fit(self, X, y):
        self.Xtr = X
        self.ytr = y

        if self.K is None:
            self.K = np.dot(X, X.T)

        [n_s, n_f] = X.shape
        if y.ndim == 1:
            y = scipy.reshape(y, (n_s, 1))

        S, U, ldelta0 = self.train_nullmodel(y, self.K, numintervals=self.numintervals,
                                             ldeltamin=self.ldeltamin, ldeltamax=self.ldeltamax,scale=self.scale,
                                             KSquare=self.KSquare, KSearch=self.KSearch)

        self.delta0 = scipy.exp(ldelta0)
        Sdi = 1. / (S + self.delta0)
        Sdi_sqrt = scipy.sqrt(Sdi)
        SUX = scipy.dot(U.T, X)
        SUX = SUX * scipy.tile(Sdi_sqrt, (n_f, 1)).T
        SUy = scipy.dot(U.T, y)
        SUy = SUy * scipy.reshape(Sdi_sqrt, (n_s, 1))

        self.clf.fit(SUX, SUy)

    def predict(self, X):
        if self.Kv is None:
            y = self.clf.predict(X) + scipy.dot(np.dot(X, self.Xtr.T), linalg.solve(self.K + self.delta0 * scipy.eye(self.Xtr.shape[0]),
                                                                                self.ytr - self.clf.predict(self.Xtr)))
        else:
            y = self.clf.predict(X) + scipy.dot(self.Kv, linalg.solve(self.K + self.delta0 * scipy.eye(self.Xtr.shape[0]),
                                                                                self.ytr - self.clf.predict(self.Xtr)))

        maxi = np.max(y)
        mini = np.min(y)
        y = (y-mini)*3/(maxi-mini)
        y = np.round(y).astype(int)
        return y

    def nLLeval(self, ldelta, Uy, S, REML=True):
        n_s = Uy.shape[0]
        delta = scipy.exp(ldelta)

        Sd = S + delta
        ldet = scipy.sum(scipy.log(Sd))

        Sdi = 1.0 / Sd
        Uy = Uy.flatten()
        ss = 1. / n_s * (Uy * Uy * Sdi).sum()

        nLL = 0.5 * (n_s * scipy.log(2.0 * scipy.pi) + ldet + n_s + n_s * scipy.log(ss))

        if REML:
            pass

        return nLL


    def train_nullmodel(self, y, K, S=None, U=None, numintervals=100, ldeltamin=-5, ldeltamax=5, scale=0, KSquare=False,
                    KSearch=False):

        ldeltamin += scale
        ldeltamax += scale

        n_s = y.shape[0]

        # rotate data
        if S is None or U is None:
            S, U = linalg.eigh(K)

        if not KSearch:
            if KSquare:
                S = np.power(S, 2)
        Uy = scipy.dot(U.T, y)

        # grid search
        if not KSearch:
            nllgrid = scipy.ones(numintervals + 1) * scipy.inf
            ldeltagrid = scipy.arange(numintervals + 1) / (numintervals * 1.0) * (ldeltamax - ldeltamin) + ldeltamin
            nllmin = scipy.inf
            for i in scipy.arange(numintervals + 1):
                nllgrid[i] = self.nLLeval(ldeltagrid[i], Uy, S)

            # find minimum
            nllmin = nllgrid.min()
            ldeltaopt_glob = ldeltagrid[nllgrid.argmin()]

            # more accurate search around the minimum of the grid search

            for i in scipy.arange(numintervals - 1) + 1:
                if (nllgrid[i] < nllgrid[i - 1] and nllgrid[i] < nllgrid[i + 1]):
                    ldeltaopt, nllopt, iter, funcalls = opt.brent(self.nLLeval, (Uy, S),
                                                                  (ldeltagrid[i - 1], ldeltagrid[i], ldeltagrid[i + 1]),
                                                                  full_output=True)
                    if nllopt < nllmin:
                        nllmin = nllopt
                        ldeltaopt_glob = ldeltaopt

        else:
            kchoices = [1., 2., 3., 4., 5.]
            knum = len(kchoices)
            global_k = scipy.inf
            global_ldeltaopt = scipy.inf
            global_min = scipy.inf
            for ki in range(knum):
                kc = kchoices[ki]
                if kc == 1:
                    Stmp = S
                elif kc.is_integer():
                    Stmp = np.power(S, kc)
                else:
                    Stmp = np.power(np.abs(S), kc)
                Uy = scipy.dot(U.T, y)
                nllgrid = scipy.ones(numintervals + 1) * scipy.inf
                ldeltagrid = scipy.arange(numintervals + 1) / (numintervals * 1.0) * (ldeltamax - ldeltamin) + ldeltamin
                nllmin = scipy.inf
                for i in scipy.arange(numintervals + 1):
                    nllgrid[i] = self.nLLeval(ldeltagrid[i], Uy, Stmp)
                nll_min = nllgrid.min()
                ldeltaopt_glob = ldeltagrid[nllgrid.argmin()]
                for i in scipy.arange(numintervals - 1) + 1:
                    if (nllgrid[i] < nllgrid[i - 1] and nllgrid[i] < nllgrid[i + 1]):
                        ldeltaopt, nllopt, iter, funcalls = opt.brent(self.nLLeval, (Uy, Stmp),
                                                                      (ldeltagrid[i - 1], ldeltagrid[i], ldeltagrid[i + 1]),
                                                                      full_output=True)
                        if nllopt < nllmin:
                            nll_min = nllopt
                            ldeltaopt_glob = ldeltaopt
                print kc, nll_min, ldeltaopt_glob
                if nll_min < global_min:
                    global_min = nll_min
                    global_ldeltaopt = ldeltaopt_glob
                    global_k = kc
            ldeltaopt_glob = global_ldeltaopt
            S = np.power(S, global_k)
            print global_k

        return S, U, ldeltaopt_glob


    def factor(self, X, rho):
        """
        computes cholesky factorization of the kernel K = 1/rho*XX^T + I

        Input:
        X design matrix: n_s x n_f (we assume n_s << n_f)
        rho: regularizaer

        Output:
        L  lower triangular matrix
        U  upper triangular matrix
        """
        n_s, n_f = X.shape
        K = 1 / rho * scipy.dot(X, X.T) + scipy.eye(n_s)
        U = linalg.cholesky(K)
        return U



