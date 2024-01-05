'''
A class for implementing EFA
'''
import numpy as np
import class_PCA as pca
import scipy.stats as sts


class EFA:

    def __init__(self, matrice):  # parametru asteptat este un numpy.ndarray
        self.X = matrice

        # intantiere model PCA
        pcaModel = pca.PCA(self.X)
        self.Xstd = pcaModel.getXstd()
        self.Corr = pcaModel.getCorr()
        self.ValProp = pcaModel.getValProp()
        self.Scoruri = pcaModel.getScoruri()
        self.CalObs = pcaModel.CalObs

    def getXstd(self):
        return self.Xstd

    def getValProp(self):
        return self.ValProp

    def getScoruri(self):
        return self.Scoruri

    def getCalObs(self):
        return self.CalObs

    def calculTestBartlett(self, loadings, epsilon):
        n = self.X.shape[0]
        m, q = np.shape(loadings)
        print(n, m, q)
        V = self.Corr
        # creare matrice diagonala de factori specifici
        psi = np.diag(epsilon)
        Vestim = loadings @ np.transpose(loadings) + psi
        Iestim = np.linalg.inv(Vestim) @ V
        detIestim = np.linalg.det(Iestim)
        if detIestim > 0: # if it aproaches positive 0 ( to can apply logarithm logic _
            traceIestim = np.trace(Iestim)
            chi2Calc = (n - 1 - (2*m - 4*q - 5) / 6) * \
                       (traceIestim - np.log(detIestim) - m)
            numarGradeLibertate = ((m - q)**2 - m - q) / 2
            chi2Tab = 1 - sts.chi2.cdf(chi2Calc, numarGradeLibertate)
        else:
            chi2Calc, chi2Tab = np.nan, np.nan # if I have these nan i stop extracting numbers

        return chi2Calc, chi2Tab


