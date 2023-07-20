import numpy
from .utils_file import mcol

def PCA(D, L, m):
    mu = D.mean(1)
    mu = mcol(mu)
    DC = D - mu
    N = D.shape[1]
    C = numpy.dot(DC,DC.T)/N
    s, U = numpy.linalg.eigh(C)
    P = U[:, ::-1][:, 0:m]
    DP = numpy.dot(P.T, D)

    return DP

def PCA_P(D, m):
    mu = D.mean(1)
    mu = mcol(mu)
    DC = D - mu
    N = D.shape[1]
    C = numpy.dot(DC,DC.T)/N
    s, U = numpy.linalg.eigh(C)
    P = U[:, ::-1][:, 0:m]

    return P

def apply_PCA(DTR, DTE, pca):
    P = PCA_P(DTR, pca)
    DTR = numpy.dot(P.T, DTR)
    DTE = numpy.dot(P.T, DTE)

    return DTR, DTE