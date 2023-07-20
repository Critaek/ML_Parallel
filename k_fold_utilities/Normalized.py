import numpy
import scipy.stats as st

path = "data/norm.npy"

def NormDTE(DTR, DTE):
    N = DTR.shape[1]

    ranks = []
    for i in range(DTR.shape[0]): #per ogni feature
        row = []
        for j in range(DTE.shape[1]): #per ogni sample
            r = rank(DTE[i,j], DTR, N, i)
            value = st.norm.ppf(r,0,1)
            row.append(value)
        ranks.append(row)
    
    mat = numpy.vstack(ranks)

    return mat #sarebbe il DTE trasformato

def rank(x, D, N, i): #Sample rank
    accum = 0
    for j in D[i, :]:
        accum += int(j<x)
    return (accum + 1) / (N + 2)

def saveNormFolds(D, L, K):
    print(f"Generating and saving normalized folds with K = {K}")
    numpy.random.seed(0)
    idx = numpy.random.permutation(D.shape[1])

    N = D.shape[1]
    M = round(N/K)

    folds = []

    for i in range(K): #K=3 -> 0,1,2
        idxTrain = numpy.concatenate([idx[0:i*M], idx[(i+1)*M:N]])
        idxTest = idx[i*M:(i+1)*M]
        DTR = D[:, idxTrain]
        LTR = L[idxTrain]
        DTE = D[:, idxTest]
        LTE = L[idxTest]
        DTR = NormDTE(DTR, DTR)
        DTE = NormDTE(DTR, DTE)
        obj = (DTR, LTR, DTE, LTE)
        folds.append(obj)

    folds = numpy.array(folds, dtype=object) #Questi oggetti hanno forma (DTR, LTR, DTE, LTE)
    numpy.save(path, folds)
    return folds

def loadNormFolds():
    raw = numpy.load(path, allow_pickle=True)

    return raw

def getNormPath():
    return path

def getSavedNormFoldsK():
    norm = numpy.load(path, allow_pickle=True)
    K = norm.shape[0]

    return K