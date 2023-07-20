import numpy

import matplotlib.pyplot as plt

from k_fold_utilities.Raw import getShuffledLabels

def PredicionsByScore(score, LLRs):
    pred = numpy.zeros(LLRs.shape)
    # HERE WE TRY TO OPTIMIZE THE THRESHOLD USING DIFFERENT VALUES FROM A SET OF TEST SCORES
    
    threshold=score
    pred = LLRs > threshold
    
    return pred

#-------------------------------------------------------------------------------------------------#

def Predictions(pi1,Cfn,Cfp, LLRs):
    pi0 = 1-pi1
    pred = numpy.zeros(LLRs.shape)
    threshold = -numpy.log((pi1*Cfn)/((pi0*Cfp)))
    #WE USE PARTICULAR VALUE OF PI IN ORDER TO COMPUTE BYAS ERROR
    for i in range(LLRs.size):
            if(LLRs[i]>threshold):
                pred[i] = 1
            else:
                pred[i] = 0
    return pred

#-------------------------------------------------------------------------------------------------#

def ConfusionMatrix(pi1,Cfn,Cfp):
    pi0=1-pi1
    commediaLLRs = numpy.load('data/commedia_llr_infpar.npy')
    pred = numpy.zeros(commediaLLRs.shape)
    threshold = -numpy.log( (pi1*Cfn) / ( (pi0*Cfp) ) )
    for i in range(commediaLLRs.size):
            if(commediaLLRs[i]>threshold):
                pred[i] = 1
            else:
                pred[i] = 0
    return pred

#-------------------------------------------------------------------------------------------------#

def BiasRisk(pi1,Cfn,Cfp,M):
    FNR = M[0][1]/(M[0][1]+M[1][1])
    FPR = M[1][0]/(M[0][0]+M[1][0])
    return (((pi1*Cfn*FNR)+(1-pi1)*Cfp*FPR),FPR,1-FNR)

#-------------------------------------------------------------------------------------------------#

def MinDummy(pi1,Cfn,Cfp):
    dummyAlwaysReject=pi1*Cfn
    dummyAlwaysAccept=(1-pi1)*Cfp
    if(dummyAlwaysReject < dummyAlwaysAccept):
        return dummyAlwaysReject
    else:
        return dummyAlwaysAccept

#-------------------------------------------------------------------------------------------------#

def assign_labels(scores, pi, Cfn, Cfp):
    threshold = - numpy.log( (pi*Cfn) / ( (pi*Cfp) ) )
    P = scores > threshold

    return numpy.int32(P)

#-------------------------------------------------------------------------------------------------#

def printDCFs(D, L, LLRs, pi_tilde):
    pi1 = pi_tilde
    pi0 = 1- pi_tilde
    Cfn = 1
    Cfp = 1
    classPriors = numpy.array([pi1,pi0]) #[0.5, 0.5]
    minDCF = []

    #normalizedDCF

    pred = assign_labels(LLRs, pi_tilde, Cfn, Cfp)
    confusionMatrix = numpy.zeros((2, 2))

    for i in range(0,len(classPriors)):
        for j in range(0,len(classPriors)):
            confusionMatrix[i,j] = ((L == j) * (pred == i)).sum()

    (DCFu,FPRi,TPRi) = BiasRisk(pi1,Cfn,Cfp,confusionMatrix)
        
    minDummy = MinDummy(pi1,Cfn,Cfp)
    ActDCF = DCFu/minDummy

    #minDCF
    comm = sorted(LLRs) #aggiungere -inf, inf

    for score in comm:
        
        Predicions_By_Score = PredicionsByScore(score, LLRs)
        labels = L
        
        confusionMatrix = numpy.zeros((2, 2))

        for i in range(0,len(classPriors)):
            for j in range(0,len(classPriors)):
                confusionMatrix[i,j] = ((labels == j) * (Predicions_By_Score == i)).sum()

        (DCFu,FPRi,TPRi) = BiasRisk(pi1,Cfn,Cfp,confusionMatrix)
        
        minDummy = MinDummy(pi1,Cfn,Cfp)
        normalizedDCF = DCFu/minDummy
        minDCF.append(normalizedDCF)

    minDCF=min(minDCF)

    return ActDCF, minDCF

#-------------------------------------------------------------------------------------------------#

def BiasErrorPlot(L, pred, scores, pi):
    piList = numpy.linspace(-4, 4, 51)
    Cfn = 1
    Cfp = 1
    pi1 = 1 - pi
    classPriors = numpy.array([pi,pi1])

    numpy.random.seed(0)
    idx = numpy.random.permutation(L.size)

    L = L[idx]

    ActDCF_List = []
    MinDCF_List = []

    for p in piList:
        pi_tilde = 1/(1+ numpy.exp(-p))
        confusionMatrix = numpy.zeros((2, 2))

        for i in range(0,len(classPriors)):
            for j in range(0,len(classPriors)):
                confusionMatrix[i,j] = ((L == j) * (pred == i)).sum()

        (DCFu,FPRi,TPRi) = BiasRisk(pi_tilde,Cfn,Cfp,confusionMatrix)
        
        minDummy = MinDummy(pi_tilde,Cfn,Cfp)
        ActDCF = DCFu/minDummy

        ActDCF_List.append(ActDCF)

    sort = sorted(scores)

    for p in piList:
        minDCF2 = []
        for score in sort:
            pi_tilde = 1/(1+ numpy.exp(-p))
            confusionMatrix = numpy.zeros((2, 2))
            Predicions_By_Score = PredicionsByScore(score, scores)

            for i in range(0,len(classPriors)):
                for j in range(0,len(classPriors)):
                    confusionMatrix[i,j] = ((L == j) * (Predicions_By_Score == i)).sum()

            (DCFu,FPRi,TPRi) = BiasRisk(pi_tilde,Cfn,Cfp,confusionMatrix)
        
            minDummy = MinDummy(pi_tilde,Cfn,Cfp)
            NormalizedDCF = DCFu/minDummy
            minDCF2.append(NormalizedDCF)
    
        MinDCF_List.append(min(minDCF2))

    return piList, ActDCF_List, MinDCF_List
