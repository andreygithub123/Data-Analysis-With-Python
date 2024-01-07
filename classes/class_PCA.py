'''
A class for implementing PCA (Principal Components Analysis)
'''

import numpy as np

class PCA:
    # assume the X is a standardised numpy.ndarray
    # need tests here
    def __init__(self, X):
        self.X = X
        # compute variance-covariaance MATRIXX of X (variance on the diagonal between same element ; other elements covariance between elements)
        # X * X transposed
        self.Cov=np.cov(m=X, rowvar=False) # we have the variables on the columns
        #print(self.Cov.shape)


        # compute eigen values / vectors for variance-covariance matrix
        self.eigenvalues, self.eigenvectors = np.linalg.eigh(a=self.Cov)
        #print(self.eigenvalues,self.eigenvalues.shape)
        #print(self.eigenvectors.shape)


        # sort the eigenvalues and eigenvector in a descending order
        k_desc = [k for  k in reversed(np.argsort(a=self.eigenvalues))] # reversed bc argsort gives ordered
        #print(k_desc)
        # we apply the list of indeces to the matrix
        self.alpha=self.eigenvalues[k_desc] # list probably
        self.A = self.eigenvectors[:,k_desc] # eigenvectors is a matrix we  want to sort the column; all the lines k_desc column
        # regularization of eigenvectors
        for j in range(self.A.shape[1]):
            minCol = np.min(a=self.A[:,j])
            maxCol = np.max(a=self.A[:,j])
            if np.abs(minCol) > np.abs(maxCol):
                #multiplying a eigenvector with a sclaar does not change the nature of an eigen vector
                self.A[:,j] = (-1) * self.A[:,j]
        #print(self.A.shape)


        # compute the principal components
        #self.C=np.matmul(self.X,self.A)
        self.C = self.X @ self.A
        # compute factor loadings
        self.Rxc = self.A * np.sqrt(self.alpha)
        # the matrix of squared principal components
        self.C2 = self.C * self.C
        #self.C2=np.squarE(self.C)


    def getEigenValues(self):
        #return self.eigenvalues
         return self.alpha

    def getEigenVector(self):
        #return self.eigenvectors
        return self.A

    def getPrinComp(self):
        return self.C

    def getFactorLoadings(self):
        return self.Rxc

    def getScores(self):
        return self.C / np.sqrt(self.alpha)

    def getQualitObs(self):
        SL = np.sum(a=self.C2, axis=1)
        return np.transpose(self.C2.T/SL)
        #return np.transpose(self.C2/SL)
        #return np.transpose(self.C2/SL[:,np.newaxis)


    def getContribObs(self):
        return self.C2 / (self.X.shape[0] * self.alpha)

    def getCommonalities(self):
        Rxc2 = np.square(self.Rxc)
        return np.cumsum(a=Rxc2, axis=1) # cummulative sums on the lines

