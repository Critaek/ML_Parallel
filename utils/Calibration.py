import numpy
from models.Regression import LogisticRegression_w_b
from k_fold_utilities.Raw import getSavedRawFoldsK

def calibrateScores(scores, labels, prior_t, seed = 42):
    l = 1e-5
    K = getSavedRawFoldsK()
    numpy.random.seed(seed)
    idx = numpy.random.permutation(scores.shape[0])

    scores = scores[idx]
    labels = labels[idx]

    N = scores.shape[0]
    M = round(N/K)

    cal_scores = []

    lab = []

    var = numpy.log(prior_t/(1-prior_t))

    for i in range(K): #K=3 -> 0,1,2
        idxTrain = numpy.concatenate([idx[0:i*M], idx[(i+1)*M:N]])
        idxTest = idx[i*M:(i+1)*M]

        scores_train = scores[idxTrain]
        scores_test = scores[idxTest]
        labels_train = labels[idxTrain]
        labels_test = labels[idxTest]

        alpha, beta = LogisticRegression_w_b(scores_train, labels_train, l)

        cal_score = alpha * scores_test + beta - var
        cal_scores.append(cal_score)
        lab.append(labels_test)

    cal_scores = numpy.concatenate(cal_scores)
    lab = numpy.concatenate(lab)

    return cal_scores, lab


