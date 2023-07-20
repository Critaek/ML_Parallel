from utils.utils_file import mcol, vrow
import numpy
import math
import utils.DimReduction as dr
from utils.DimReduction import apply_PCA
from k_fold_utilities.Raw import loadRawFolds
from k_fold_utilities.Normalized import loadNormFolds
import utils.ModelEvaluation as me
from typing import List, Optional
from utils.utils_file import load_test, load_norm_test, load_train, load_norm_train

def meanAndCovMat(X):
    N = X.shape[1]
    mu = X.mean(1) #Calcolo la media nella direzione delle colonne, quindi da sinistra verso destra
    mu = mcol(mu)
    XC = X - mu
    C = (1/N) * numpy.dot( (XC), (XC).T )
    return mu, C

def logpdf_GAU_1Sample(x, mu, C):
    #C qua rappresenta la matrice delle covarianze chiamata sigma nelle slide
    inv = numpy.linalg.inv(C)
    sign, det = numpy.linalg.slogdet(C)
    M = x.shape[0]
    ret = -(M/2) * math.log(2*math.pi) - (0.5) * det - (0.5) * numpy.dot( (x-mu).T, numpy.dot(inv, (x-mu)))
    return ret.ravel()

def logpdf_GAU_ND(X, mu, C):
    Y = [logpdf_GAU_1Sample(X[:, i:i+1], mu, C) for i in range(X.shape[1])]
    return numpy.array(Y).ravel()

class MultiVariate(object):
    def __init__(self, D, L, pca: Optional[List[int]] = None, flag: Optional[bool] = True):
        self.D = D
        self.L = L
        self.type = "Full"
        self.raw = loadRawFolds()
        self.normalized = loadNormFolds()
        if pca is None:
            self.pca = [D.shape[0]]
        else:
            assert max(pca) <= D.shape[0], f"pca must be smaller than {D.shape[0]}"
            self.pca = pca
        self.print_flag = flag
        self.print_file = "data/Results/MVG_Full.txt"

    def MultiV(self, DTR, LTR, DTE):
        mu0, C0 = meanAndCovMat(DTR[:, LTR == 0]) #Calcolo media e matrice delle covarianze per ogni classe
        mu1, C1 = meanAndCovMat(DTR[:, LTR == 1])
        S0 = logpdf_GAU_ND(DTE, mu0, C0)
        S1 = logpdf_GAU_ND(DTE, mu1, C1)
        LLRs = S1 - S0
    
        return LLRs
    
    def kFold(self, folds, pca):
        LLRs = []
        Predictions = []

        for f in folds:
            DTR = f[0]
            LTR = f[1]
            DTE = f[2]
            LTE = f[3]
            P = dr.PCA_P(DTR, pca)
            DTR = numpy.dot(P.T, DTR)
            DTE = numpy.dot(P.T, DTE)
            LLRsRet = self.MultiV(DTR, LTR, DTE)
            LLRs.append(LLRsRet)
    
        LLRs = numpy.hstack(LLRs)

        return LLRs
    
    def train(self):
        prior_tilde_set = [0.1, 0.5]

        f = open(self.print_file, 'w')

        for i in self.pca:
            LLRs = self.kFold(self.raw, i)
            for prior_tilde in prior_tilde_set:
                ActDCF, minDCF = me.printDCFs(self.D, self.L, LLRs, prior_tilde)
                if self.print_flag:  
                    print(f"{prior_tilde} | {self.type} | Raw | PCA = {i} | actDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF, 3)}")
                print(f"{prior_tilde} | {self.type} | Raw | PCA = {i} | actDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF, 3)}", file=f)

        for i in self.pca:
            LLRs = self.kFold(self.normalized, i)
            for prior_tilde in prior_tilde_set:
                ActDCF, minDCF = me.printDCFs(self.D, self.L, LLRs, prior_tilde)  
                if self.print_flag:
                    print(f"{prior_tilde} | {self.type} | Normalized | PCA = {i} | actDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF, 3)}")
                print(f"{prior_tilde} | {self.type} | Normalized | PCA = {i} | actDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF, 3)}", file=f)

    def evaluate(self):
        prior_tilde_set = [0.1, 0.5]

        file_path = "data/FinalEvaluation/MVG_Full.txt"

        f = open(file_path, "w")

        D_test, L_test = load_test()
        norm_D_test, norm_L_test = load_norm_test()
        D, L = load_train()
        norm_D, L = load_norm_train()

        for i in self.pca:
            D_pca, D_test_pca = apply_PCA(D, D_test, i)
            LLRs = self.MultiV(D_pca, L, D_test_pca)
            for prior_tilde in prior_tilde_set:
                ActDCF, minDCF = me.printDCFs(D_test_pca, L_test, LLRs, prior_tilde)
                if self.print_flag:  
                    print(f"{prior_tilde} | {self.type} | Raw | PCA = {i} | actDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF, 3)}")
                print(f"{prior_tilde} | {self.type} | Raw | PCA = {i} | actDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF, 3)}", file=f)

        for i in self.pca:
            norm_D_pca, norm_D_test_pca = apply_PCA(norm_D, norm_D_test, i)
            LLRs = self.MultiV(norm_D_pca, L, norm_D_test_pca)
            for prior_tilde in prior_tilde_set:
                ActDCF, minDCF = me.printDCFs(norm_D_test_pca, norm_L_test, LLRs, prior_tilde)  
                if self.print_flag:
                    print(f"{prior_tilde} | {self.type} | Normalized | PCA = {i} | actDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF, 3)}")
                print(f"{prior_tilde} | {self.type} | Normalized | PCA = {i} | actDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF, 3)}", file=f)



class Tied(object):
    def __init__(self, D, L, pca: Optional[List[int]] = None, flag: Optional[bool] = True):
        self.D = D
        self.L = L
        self.type = "Tied"
        self.raw = loadRawFolds()
        self.normalized = loadNormFolds()
        if pca is None:
            self.pca = [D.shape[0]]
        else:
            assert max(pca) <= D.shape[0], f"pca must be smaller than {D.shape[0]}"
            self.pca = pca
        self.print_flag = flag
        self.print_file = "data/Results/MVG_Tied.txt"

    def Tied(self, DTR, LTR, DTE):
        mu0, C0 = meanAndCovMat(DTR[:, LTR == 0]) #Calcolo media e matrice delle covarianze per ogni classe
        mu1, C1 = meanAndCovMat(DTR[:, LTR == 1])
        N0 = DTR[:, LTR == 0].shape[1]            #Queste sarebbero le mie Nc, dove a c è sostituito il numero della classe
        N1 = DTR[:, LTR == 1].shape[1]
        N = DTR.shape[1]                          #Prendo la grandezza del mio traning set, quindi quanti sample contiene
        nC0 = N0*C0
        nC1 = N1*C1
        C = numpy.add(nC0, nC1)
        C = C/N

        S0 = logpdf_GAU_ND(DTE, mu0, C)
        S1 = logpdf_GAU_ND(DTE, mu1, C)
        LLRs = S1 - S0

        return LLRs
    
    def kFold(self, folds, pca):
        LLRs = []
        Predictions = []

        for f in folds:
            DTR = f[0]
            LTR = f[1]
            DTE = f[2]
            LTE = f[3]
            P = dr.PCA_P(DTR, pca)
            DTR = numpy.dot(P.T, DTR)
            DTE = numpy.dot(P.T, DTE)
            LLRsRet = self.Tied(DTR, LTR, DTE)
            LLRs.append(LLRsRet)
    
        LLRs = numpy.hstack(LLRs)

        return LLRs
    
    def train(self):
        prior_tilde_set = [0.1, 0.5]

        f = open(self.print_file, 'w')

        for i in self.pca:
            LLRs = self.kFold(self.raw, i)
            for prior_tilde in prior_tilde_set:
                ActDCF, minDCF = me.printDCFs(self.D, self.L, LLRs, prior_tilde)  
                if self.print_flag:
                    print(f"{prior_tilde} | {self.type} | Raw | PCA = {i} | actDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF, 3)}")
                print(f"{prior_tilde} | {self.type} | Raw | PCA = {i} | actDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF, 3)}", file=f)

        for i in self.pca:
            LLRs = self.kFold(self.normalized, i)
            for prior_tilde in prior_tilde_set:
                ActDCF, minDCF = me.printDCFs(self.D, self.L, LLRs, prior_tilde)
                if self.print_flag:
                    print(f"{prior_tilde} | {self.type} | Normalized | PCA = {i} | actDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF, 3)}")
                print(f"{prior_tilde} | {self.type} | Normalized | PCA = {i} | actDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF, 3)}", file=f)
    
    def evaluate(self):
        prior_tilde_set = [0.1, 0.5]

        file_path = "data/FinalEvaluation/MVG_Tied.txt"

        f = open(file_path, "w")

        D_test, L_test = load_test()
        norm_D_test, norm_L_test = load_norm_test()
        D, L = load_train()
        norm_D, L = load_norm_train()

        for i in self.pca:
            D_pca, D_test_pca = apply_PCA(D, D_test, i)
            LLRs = self.Tied(D_pca, L, D_test_pca)
            for prior_tilde in prior_tilde_set:
                ActDCF, minDCF = me.printDCFs(D_test_pca, L_test, LLRs, prior_tilde)
                if self.print_flag:  
                    print(f"{prior_tilde} | {self.type} | Raw | PCA = {i} | actDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF, 3)}")
                print(f"{prior_tilde} | {self.type} | Raw | PCA = {i} | actDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF, 3)}", file=f)

        for i in self.pca:
            norm_D_pca, norm_D_test_pca = apply_PCA(norm_D, norm_D_test, i)
            LLRs = self.Tied(norm_D_pca, L, norm_D_test_pca)
            for prior_tilde in prior_tilde_set:
                ActDCF, minDCF = me.printDCFs(norm_D_test_pca, norm_L_test, LLRs, prior_tilde)  
                if self.print_flag:
                    print(f"{prior_tilde} | {self.type} | Normalized | PCA = {i} | actDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF, 3)}")
                print(f"{prior_tilde} | {self.type} | Normalized | PCA = {i} | actDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF, 3)}", file=f)

class Bayes(object):
    def __init__(self, D, L, pca: Optional[List[int]] = None, flag: Optional[bool] = True):
        self.D = D
        self.L = L
        self.type = "Bayes"
        self.raw = loadRawFolds()
        self.normalized = loadNormFolds()
        if pca is None:
            self.pca = [D.shape[0]]
        else:
            assert max(pca) <= D.shape[0], f"pca must be smaller than {D.shape[0]}"
            self.pca = pca
        self.print_flag = flag
        self.print_file = "data/Results/MVG_Bayes.txt"

    def Bayes(self, DTR, LTR, DTE):
        mu0, C0 = meanAndCovMat(DTR[:, LTR == 0]) #Calcolo media e matrice delle covarianze per ogni classe
        mu1, C1 = meanAndCovMat(DTR[:, LTR == 1])
        I = numpy.identity(C0.shape[0])
        C0Diag = C0 * I
        C1Diag = C1 * I
        S0 = logpdf_GAU_ND(DTE, mu0, C0Diag)
        S1 = logpdf_GAU_ND(DTE, mu1, C1Diag)
        
        LLRs = S1 - S0

        return LLRs
    
    def kFold(self, folds, pca):
        LLRs = []
        Predictions = []

        for f in folds:
            DTR = f[0]
            LTR = f[1]
            DTE = f[2]
            LTE = f[3]
            P = dr.PCA_P(DTR, pca)
            DTR = numpy.dot(P.T, DTR)
            DTE = numpy.dot(P.T, DTE)
            LLRsRet = self.Bayes(DTR, LTR, DTE)
            LLRs.append(LLRsRet)
    
        LLRs = numpy.hstack(LLRs)

        return LLRs
    
    def train(self):
        prior_tilde_set = [0.1, 0.5]

        f = open(self.print_file, 'w')

        for i in self.pca:
            LLRs = self.kFold(self.raw, i)
            for prior_tilde in prior_tilde_set:
                ActDCF, minDCF = me.printDCFs(self.D, self.L, LLRs, prior_tilde)
                if self.print_flag:  
                    print(f"{prior_tilde} | {self.type} | Raw | PCA = {i} | actDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF, 3)}")
                print(f"{prior_tilde} | {self.type} | Raw | PCA = {i} | actDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF, 3)}", file=f)

        for i in self.pca:
            LLRs = self.kFold(self.normalized, i)
            for prior_tilde in prior_tilde_set:
                ActDCF, minDCF = me.printDCFs(self.D, self.L, LLRs, prior_tilde) 
                if self.print_flag: 
                    print(f"{prior_tilde} | {self.type} | Normalized | PCA = {i} | actDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF, 3)}")
                print(f"{prior_tilde} | {self.type} | Normalized | PCA = {i} | actDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF, 3)}", file=f)

    def evaluate(self):
        prior_tilde_set = [0.1, 0.5]

        file_path = "data/FinalEvaluation/MVG_Bayes.txt"

        f = open(file_path, "w")

        D_test, L_test = load_test()
        norm_D_test, norm_L_test = load_norm_test()
        D, L = load_train()
        norm_D, L = load_norm_train()

        for i in self.pca:
            D_pca, D_test_pca = apply_PCA(D, D_test, i)
            LLRs = self.Bayes(D_pca, L, D_test_pca)
            for prior_tilde in prior_tilde_set:
                ActDCF, minDCF = me.printDCFs(D_test_pca, L_test, LLRs, prior_tilde)
                if self.print_flag:  
                    print(f"{prior_tilde} | {self.type} | Raw | PCA = {i} | actDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF, 3)}")
                print(f"{prior_tilde} | {self.type} | Raw | PCA = {i} | actDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF, 3)}", file=f)

        for i in self.pca:
            norm_D_pca, norm_D_test_pca = apply_PCA(norm_D, norm_D_test, i)
            LLRs = self.Bayes(norm_D_pca, L, norm_D_test_pca)
            for prior_tilde in prior_tilde_set:
                ActDCF, minDCF = me.printDCFs(norm_D_test_pca, norm_L_test, LLRs, prior_tilde)  
                if self.print_flag:
                    print(f"{prior_tilde} | {self.type} | Normalized | PCA = {i} | actDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF, 3)}")
                print(f"{prior_tilde} | {self.type} | Normalized | PCA = {i} | actDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF, 3)}", file=f)