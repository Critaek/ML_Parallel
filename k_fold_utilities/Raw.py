import numpy

path = "data/raw.npy"

def saveRawFolds(D, L, K):
    print(f"Generating and saving raw folds with K = {K}")
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
        obj = (DTR, LTR, DTE, LTE)
        folds.append(obj)

    folds = numpy.array(folds, dtype=object)
    numpy.save(path, folds)

def loadRawFolds():
    raw = numpy.load(path, allow_pickle=True)

    return raw

def getRawPath():
    return path

def getSavedRawFoldsK():
    raw = numpy.load(path, allow_pickle=True)
    K = raw.shape[0]

    return K

def getShuffledLabels():
    raw = numpy.load(path, allow_pickle=True)

    labels = []

    for f in raw:
        LTE = f[3]
        labels.append(LTE)

    return numpy.concatenate(labels)
