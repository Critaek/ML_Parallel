from k_fold_utilities.Raw import loadRawFolds
from k_fold_utilities.Normalized import loadNormFolds
from typing import Optional, List
import numpy
import scipy
import math
import utils.DimReduction as dr
from utils.utils_file import vrow, mcol
import utils.ModelEvaluation as me
from utils.Plot import plotHistGMM
from tqdm import tqdm
from utils.Calibration import calibrateScores
from utils.utils_file import load_test, load_norm_test, load_train, load_norm_train
from utils.DimReduction import apply_PCA

def meanAndCovMat(X):
    N = X.shape[1]
    mu = X.mean(1) #calcolo la media nella direzione delle colonne, quindi da sinistra verso destra
    mu = mcol(mu)
    XC = X - mu
    C = (1/N) * numpy.dot( (XC), (XC).T )
    return mu, C

def GMM_ll_perSample(X, gmm):

    G = len(gmm)
    N = X.shape[1]
    S = numpy.zeros((G, N))
    for g in range(G):
        S[g, :] = logpdf_GAU_ND_Opt(X, gmm[g][1], gmm[g][2]) + numpy.log(gmm[g][0])
    
    return scipy.special.logsumexp(S, axis = 0)

def logpdf_GAU_ND_Opt(X, mu, C):
    inv = numpy.linalg.inv(C)
    sign, det = numpy.linalg.slogdet(C)
    M = X.shape[0]
    const = -(M/2) * math.log(2*math.pi) - (0.5) * det 
    
    Y = []
    for i in range(X.shape[1]):
        x = X[:, i:i+1]
        res = const - (0.5) * numpy.dot( (x-mu).T, numpy.dot(inv, (x-mu)))
        Y.append(res)

    return numpy.array(Y).ravel()

def GMM_Scores(DTE, gmm0, gmm1):
    Scores0 = GMM_ll_perSample(DTE, gmm0)
    Scores1 = GMM_ll_perSample(DTE, gmm1)
    
    Scores = Scores1 - Scores0

    return Scores

class GMMFull(object):
    def __init__(self, D, L, n_Set, pca: Optional[List[int]] = None, flag: Optional[bool] = True):
        self.D = D
        self.L = L
        self.type = "GMM Full"
        self.raw = loadRawFolds()
        self.normalized = loadNormFolds()
        self.n_Set = n_Set
        if pca is None:
            self.pca = [D.shape[0]]
        else:
            assert max(pca) <= D.shape[0], f"pca must be smaller than {D.shape[0]}"
            self.pca = pca
        self.print_flag = flag
        self.print_file = "data/Results_PCA5/Results/GMMFull.txt"

    def GMM_EM_Full(self, X, gmm, psi = 0.01):
        llNew = None
        llOld = None
        G = len(gmm)
        N = X.shape[1]

        while llOld is None or llNew - llOld > 1e-6:
            llOld = llNew
            SJ = numpy.zeros((G, N))
            for g in range(G): #numero componenti
                SJ[g, :] = logpdf_GAU_ND_Opt(X, gmm[g][1], gmm[g][2]) + numpy.log(gmm[g][0])
            SM = scipy.special.logsumexp(SJ, axis = 0)
            llNew = SM.sum() / N
            P = numpy.exp(SJ - SM)
            gmmNew = []

            for g in range(G):
                gamma = P[g, :]
                Z = gamma.sum()            
                F = (vrow(gamma)*X).sum(1)
                S = numpy.dot(X, (vrow(gamma)*X).T)
                w = Z / N
                mu = mcol(F / Z)
                Sigma = S/Z - numpy.dot(mu, mu.T)
                U, s, _ = numpy.linalg.svd(Sigma)
                s[s < psi] = psi
                Sigma = numpy.dot(U, mcol(s) * U.T)
                gmmNew.append((w, mu, Sigma))

            gmm = gmmNew

        return gmm

    def GMM_LBG_Full(self, X, G, alpha = 0.1):
        mu, C = meanAndCovMat(X)
        gmms = []
        
        gmms.append((1.0, mu, C))
        
        gmms = self.GMM_EM_Full(X, gmms)

        for g in range(G): #G = 2 -> 0, 1
            newList = []
            for element in gmms:
                w = element[0] / 2
                mu = element[1]
                C = element[2]
                U, s, Vh = numpy.linalg.svd(C)
                d = U[:, 0:1] * s[0]**0.5 * alpha
                newList.append((w, mu + d, C))
                newList.append((w, mu - d, C))
            gmms = self.GMM_EM_Full(X, newList)  

        return gmms 
    
    def kFold(self, folds, n, pca): #KModel è il K relativo al modello
        LLRs = []

        for f in folds:
            DTR = f[0]
            LTR = f[1]
            DTE = f[2]
            LTE = f[3]
            P = dr.PCA_P(DTR, pca)
            DTR = numpy.dot(P.T, DTR)
            DTE = numpy.dot(P.T, DTE)
            DTR0 = DTR[:, LTR == 0] # bad wines
            DTR1 = DTR[:, LTR == 1] # good wines
            gmm0 = self.GMM_LBG_Full(DTR0, n) #n number of components
            gmm1 = self.GMM_LBG_Full(DTR1, n) #n number of components
            LLRsRet = GMM_Scores(DTE, gmm0, gmm1)
            LLRs.append(LLRsRet)
        
        LLRs = numpy.hstack(LLRs)

        return LLRs
    
    def calculate_scores(self, DTR, LTR, DTE, n):
        DTR0 = DTR[:, LTR == 0]
        DTR1 = DTR[:, LTR == 1]
        gmm0 = self.GMM_LBG_Full(DTR0, n)
        gmm1 = self.GMM_LBG_Full(DTR1, n)
        LLRs = GMM_Scores(DTE, gmm0, gmm1)

        return LLRs
    
    def plot(self, flag: Optional[bool] = True):
        print("Plotting GMM Full Results...")
        f = open(self.print_file, "r")

        normalized = []
        raw = []

        for line in f:
            elements = line.split("|")
            elements =[elem.strip() for elem in elements]
            if float(float(elements[0]) == 0.5 and elements[4] == "Uncalibrated"):
                if elements[3] == "Raw":
                    raw.append(float(elements[7][9:]))
                if elements[3] == "Normalized":
                    normalized.append(float(elements[7][9:]))

        save_file = f"data/Plots/GMMFull.png"

        plotHistGMM(self.n_Set, raw, normalized, self.type, filename=save_file, flag=flag)
    
    def train(self):
        prior_tilde_set = [0.1, 0.5]

        f = open(self.print_file, "w")

        hyperparameter_list = [(int(math.log2(n)), i) for n in self.n_Set for i in self.pca]

        for n, i in tqdm(hyperparameter_list, desc="Training GMM Full...", ncols=100):
            Scores = self.kFold(self.raw, n, i)
            for prior_tilde in prior_tilde_set: 
                CalibratedScores, labels = calibrateScores(Scores, self.L, prior_tilde)
                ActDCF, minDCF = me.printDCFs(self.D, self.L, Scores, prior_tilde)
                if self.print_flag:
                    print(f"{prior_tilde} | {self.type} | nComponents = {2**n} | Raw | Uncalibrated | PCA = {i}" + \
                          f" | ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}")
                print(f"{prior_tilde} | {self.type} | nComponents = {2**n} | Raw | Uncalibrated | PCA = {i}" + \
                      f" | ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}", file=f)
                ActDCF, minDCF = me.printDCFs(self.D, labels, CalibratedScores, prior_tilde)
                if self.print_flag:
                    print(f"{prior_tilde} | {self.type} | nComponents = {2**n} | Raw | Calibrated | PCA = {i}" + \
                          f" | ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}")
                print(f"{prior_tilde} | {self.type} | nComponents = {2**n} | Raw | Calibrated | PCA = {i}" + \
                      f" | ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}", file=f)
            
            Scores = self.kFold(self.normalized, n, i)
            #Still called LLRs in the printDCFs function, but they are scores with no probabilistic interpretation
            #We use the same function for every model
            for prior_tilde in prior_tilde_set: 
                CalibratedScores, labels = calibrateScores(Scores, self.L, prior_tilde)
                ActDCF, minDCF = me.printDCFs(self.D, self.L, Scores, prior_tilde)
                if self.print_flag:
                    print(f"{prior_tilde} | {self.type} | nComponents = {2**n} | Normalized | Uncalibrated | PCA = {i}" + \
                          f" | ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}")
                print(f"{prior_tilde} | {self.type} | nComponents = {2**n} | Normalized | Uncalibrated | PCA = {i}" + \
                      f" | ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}", file=f)
                ActDCF, minDCF = me.printDCFs(self.D, labels, CalibratedScores, prior_tilde)
                if self.print_flag:
                    print(f"{prior_tilde} | {self.type} | nComponents = {2**n} | Normalized | Calibrated | PCA = {i}" + \
                          f" | ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}")
                print(f"{prior_tilde} | {self.type} | nComponents = {2**n} | Normalized | Calibrated | PCA = {i}" + \
                      f" | ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}", file=f)
                
    def evaluate(self):
        prior_tilde_set = [0.1, 0.5]

        file_path = "data/FinalEvaluation/GMMFull.txt"

        f = open(file_path, "w")

        D_test, L_test = load_test()
        norm_D_test, norm_L_test = load_norm_test()
        D, L = load_train()
        norm_D, L = load_norm_train()

        hyperparameter_list = [(int(math.log2(n)), i) for n in self.n_Set for i in self.pca]

        for n, i in tqdm(hyperparameter_list, desc="Evaluating GMM Full...", ncols=100):
            D_pca, D_test_pca = apply_PCA(D, D_test, i)
            Scores = self.calculate_scores(D_pca, L, D_test_pca, n)
            for prior_tilde in prior_tilde_set: 
                CalibratedScores, labels = calibrateScores(Scores, L_test, prior_tilde)
                ActDCF, minDCF = me.printDCFs(D_test_pca, L_test, Scores, prior_tilde)
                if self.print_flag:
                    print(f"{prior_tilde} | {self.type} | nComponents = {2**n} | Raw | Uncalibrated | PCA = {i}" + \
                          f" | ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}")
                print(f"{prior_tilde} | {self.type} | nComponents = {2**n} | Raw | Uncalibrated | PCA = {i}" + \
                      f" | ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}", file=f)
                
                ActDCF, minDCF = me.printDCFs(D_test_pca, labels, CalibratedScores, prior_tilde)
                if self.print_flag:
                    print(f"{prior_tilde} | {self.type} | nComponents = {2**n} | Raw | Calibrated | PCA = {i}" + \
                          f" | ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}")
                print(f"{prior_tilde} | {self.type} | nComponents = {2**n} | Raw | Calibrated | PCA = {i}" + \
                      f" | ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}", file=f)
            
            norm_D_pca, norm_D_test_pca = apply_PCA(norm_D, norm_D_test, i)
            Scores = self.calculate_scores(norm_D_pca, L, norm_D_test_pca, n)
            #Still called LLRs in the printDCFs function, but they are scores with no probabilistic interpretation
            #We use the same function for every model
            for prior_tilde in prior_tilde_set: 
                CalibratedScores, labels = calibrateScores(Scores, norm_L_test, prior_tilde)
                ActDCF, minDCF = me.printDCFs(norm_D_test_pca, norm_L_test, Scores, prior_tilde)
                if self.print_flag:
                    print(f"{prior_tilde} | {self.type} | nComponents = {2**n} | Normalized | Uncalibrated | PCA = {i}" + \
                          f" | ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}")
                print(f"{prior_tilde} | {self.type} | nComponents = {2**n} | Normalized | Uncalibrated | PCA = {i}" + \
                      f" | ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}", file=f)
                
                ActDCF, minDCF = me.printDCFs(norm_D_test_pca, labels, CalibratedScores, prior_tilde)
                if self.print_flag:
                    print(f"{prior_tilde} | {self.type} | nComponents = {2**n} | Normalized | Calibrated | PCA = {i}" + \
                          f" | ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}")
                print(f"{prior_tilde} | {self.type} | nComponents = {2**n} | Normalized | Calibrated | PCA = {i}" + \
                      f" | ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}", file=f)


class GMMDiagonal(object):
    def __init__(self, D, L, n_Set, pca: Optional[List[int]] = None, flag: Optional[bool] = True):
        self.D = D
        self.L = L
        self.type = "GMM Diagonal"
        self.raw = loadRawFolds()
        self.normalized = loadNormFolds()
        self.n_Set = n_Set
        if pca is None:
            self.pca = [D.shape[0]]
        else:
            assert max(pca) <= D.shape[0], f"pca must be smaller than {D.shape[0]}"
            self.pca = pca
        self.print_flag = flag
        self.print_file = "data/Results_PCA5/Results/GMMDiagonal.txt"

    def GMM_EM_Diagonal(self, X, gmm, psi = 0.01):
        llNew = None
        llOld = None
        G = len(gmm)
        N = X.shape[1]

        while llOld is None or llNew - llOld > 1e-6:
            llOld = llNew
            SJ = numpy.zeros((G, N))
            for g in range(G): #numero componenti
                SJ[g, :] = logpdf_GAU_ND_Opt(X, gmm[g][1], gmm[g][2]) + numpy.log(gmm[g][0])
            SM = scipy.special.logsumexp(SJ, axis = 0)
            llNew = SM.sum() / N
            P = numpy.exp(SJ - SM)
            gmmNew = []

            for g in range(G):
                gamma = P[g, :]
                Z = gamma.sum()            
                F = (vrow(gamma)*X).sum(1)
                S = numpy.dot(X, (vrow(gamma)*X).T)
                w = Z / N
                mu = mcol(F / Z)
                Sigma = S/Z - numpy.dot(mu, mu.T)
                U, s, _ = numpy.linalg.svd(Sigma)
                s[s < psi] = psi
                Sigma = numpy.dot(U, mcol(s) * U.T)
                
                #Diagonalizzo
                Sigma = Sigma * numpy.eye(Sigma.shape[0])

                gmmNew.append((w, mu, Sigma))

            gmm = gmmNew

        return gmm
    
    def GMM_LBG_Diagonal(self, X, G, alpha = 0.1):
        mu, C = meanAndCovMat(X)
        gmms = []
        
        gmms.append((1.0, mu, C))
        
        gmms = self.GMM_EM_Diagonal(X, gmms)

        for g in range(G): #G = 2 -> 0, 1
            newList = []
            for element in gmms:
                w = element[0] / 2
                mu = element[1]
                C = element[2]
                U, s, Vh = numpy.linalg.svd(C)
                d = U[:, 0:1] * s[0]**0.5 * alpha
                newList.append((w, mu + d, C))
                newList.append((w, mu - d, C))
            gmms = self.GMM_EM_Diagonal(X, newList)  

        return gmms 
    
    def kFold(self, folds, n, pca): #KModel è il K relativo al modello
        LLRs = []

        for f in folds:
            DTR = f[0]
            LTR = f[1]
            DTE = f[2]
            LTE = f[3]
            P = dr.PCA_P(DTR, pca)
            DTR = numpy.dot(P.T, DTR)
            DTE = numpy.dot(P.T, DTE)
            DTR0 = DTR[:, LTR == 0] # bad wines
            DTR1 = DTR[:, LTR == 1] # good wines
            gmm0 = self.GMM_LBG_Diagonal(DTR0, n) #n number of components
            gmm1 = self.GMM_LBG_Diagonal(DTR1, n) #n number of components
            LLRsRet = GMM_Scores(DTE, gmm0, gmm1)
            LLRs.append(LLRsRet)
        
        LLRs = numpy.hstack(LLRs)

        return LLRs
    
    def calculate_scores(self, DTR, LTR, DTE, n):
        DTR0 = DTR[:, LTR == 0]
        DTR1 = DTR[:, LTR == 1]
        gmm0 = self.GMM_LBG_Diagonal(DTR0, n)
        gmm1 = self.GMM_LBG_Diagonal(DTR1, n)
        LLRs = GMM_Scores(DTE, gmm0, gmm1)

        return LLRs
    
    def plot(self, flag: Optional[bool] = True):
        print("Plotting GMM Diagonal Results...")
        f = open(self.print_file, "r")

        normalized = []
        raw = []

        for line in f:
            elements = line.split("|")
            elements =[elem.strip() for elem in elements]
            if float(float(elements[0]) == 0.5 and elements[4] == "Uncalibrated"):
                if elements[3] == "Raw":
                    raw.append(float(elements[7][9:]))
                if elements[3] == "Normalized":
                    normalized.append(float(elements[7][9:]))

        save_file = f"data/Plots/GMMDiagonal.png"

        plotHistGMM(self.n_Set, raw, normalized, self.type, filename=save_file, flag=flag)
    
    def train(self):
        prior_tilde_set = [0.1, 0.5]

        f = open(self.print_file, "w")

        hyperparameter_list = [(int(math.log2(n)), i) for n in self.n_Set for i in self.pca]

        for n, i in tqdm(hyperparameter_list, desc="Training GMM Diagonal...", ncols=100):
            Scores = self.kFold(self.raw, n, i)
            for prior_tilde in prior_tilde_set: 
                CalibratedScores, labels = calibrateScores(Scores, self.L, prior_tilde)
                ActDCF, minDCF = me.printDCFs(self.D, self.L, Scores, prior_tilde)
                if self.print_flag:
                    print(f"{prior_tilde} | {self.type} | nComponents = {2**n} | Raw | Uncalibrated | PCA = {i}" + \
                          f" | ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}")
                print(f"{prior_tilde} | {self.type} | nComponents = {2**n} | Raw | Uncalibrated | PCA = {i}" + \
                      f" | ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}", file=f)
                ActDCF, minDCF = me.printDCFs(self.D, labels, CalibratedScores, prior_tilde)
                if self.print_flag:
                    print(f"{prior_tilde} | {self.type} | nComponents = {2**n} | Raw | Calibrated | PCA = {i}" + \
                          f" | ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}")
                print(f"{prior_tilde} | {self.type} | nComponents = {2**n} | Raw | Calibrated | PCA = {i}" + \
                      f" | ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}", file=f)
            
            Scores = self.kFold(self.normalized, n, i)
            for prior_tilde in prior_tilde_set: 
                CalibratedScores, labels = calibrateScores(Scores, self.L, prior_tilde)
                ActDCF, minDCF = me.printDCFs(self.D, self.L, Scores, prior_tilde)
                if self.print_flag:
                    print(f"{prior_tilde} | {self.type} | nComponents = {2**n} | Normalized | Uncalibrated | PCA = {i}" + \
                          f" | ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}")
                print(f"{prior_tilde} | {self.type} | nComponents = {2**n} | Normalized | Uncalibrated | PCA = {i}" + \
                      f" | ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}", file=f)
                ActDCF, minDCF = me.printDCFs(self.D, labels, CalibratedScores, prior_tilde)
                if self.print_flag:
                    print(f"{prior_tilde} | {self.type} | nComponents = {2**n} | Normalized | Calibrated | PCA = {i}" + \
                          f" | ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}")
                print(f"{prior_tilde} | {self.type} | nComponents = {2**n} | Normalized | Calibrated | PCA = {i}" + \
                      f" | ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}", file=f)

    def evaluate(self):
        prior_tilde_set = [0.1, 0.5]

        file_path = "data/FinalEvaluation/GMMDiagonal.txt"

        f = open(file_path, "w")

        D_test, L_test = load_test()
        norm_D_test, norm_L_test = load_norm_test()
        D, L = load_train()
        norm_D, L = load_norm_train()

        hyperparameter_list = [(int(math.log2(n)), i) for n in self.n_Set for i in self.pca]

        for n, i in tqdm(hyperparameter_list, desc="Evaluating GMM Diagonal...", ncols=100):
            D_pca, D_test_pca = apply_PCA(D, D_test, i)
            Scores = self.calculate_scores(D_pca, L, D_test_pca, n)
            for prior_tilde in prior_tilde_set: 
                CalibratedScores, labels = calibrateScores(Scores, L_test, prior_tilde)
                ActDCF, minDCF = me.printDCFs(D_test_pca, L_test, Scores, prior_tilde)
                if self.print_flag:
                    print(f"{prior_tilde} | {self.type} | nComponents = {2**n} | Raw | Uncalibrated | PCA = {i}" + \
                          f" | ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}")
                print(f"{prior_tilde} | {self.type} | nComponents = {2**n} | Raw | Uncalibrated | PCA = {i}" + \
                      f" | ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}", file=f)
                
                ActDCF, minDCF = me.printDCFs(D_test_pca, labels, CalibratedScores, prior_tilde)
                if self.print_flag:
                    print(f"{prior_tilde} | {self.type} | nComponents = {2**n} | Raw | Calibrated | PCA = {i}" + \
                          f" | ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}")
                print(f"{prior_tilde} | {self.type} | nComponents = {2**n} | Raw | Calibrated | PCA = {i}" + \
                      f" | ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}", file=f)
            
            norm_D_pca, norm_D_test_pca = apply_PCA(norm_D, norm_D_test, i)
            Scores = self.calculate_scores(norm_D_pca, L, norm_D_test_pca, n)
            #Still called LLRs in the printDCFs function, but they are scores with no probabilistic interpretation
            #We use the same function for every model
            for prior_tilde in prior_tilde_set: 
                CalibratedScores, labels = calibrateScores(Scores, norm_L_test, prior_tilde)
                ActDCF, minDCF = me.printDCFs(norm_D_test_pca, norm_L_test, Scores, prior_tilde)
                if self.print_flag:
                    print(f"{prior_tilde} | {self.type} | nComponents = {2**n} | Normalized | Uncalibrated | PCA = {i}" + \
                          f" | ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}")
                print(f"{prior_tilde} | {self.type} | nComponents = {2**n} | Normalized | Uncalibrated | PCA = {i}" + \
                      f" | ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}", file=f)
                
                ActDCF, minDCF = me.printDCFs(norm_D_test_pca, labels, CalibratedScores, prior_tilde)
                if self.print_flag:
                    print(f"{prior_tilde} | {self.type} | nComponents = {2**n} | Normalized | Calibrated | PCA = {i}" + \
                          f" | ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}")
                print(f"{prior_tilde} | {self.type} | nComponents = {2**n} | Normalized | Calibrated | PCA = {i}" + \
                      f" | ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}", file=f)

class GMMTied(object):
    def __init__(self, D, L, n_Set, pca: Optional[List[int]] = None, flag: Optional[bool] = True):
        self.D = D
        self.L = L
        self.type = "GMM Tied"
        self.raw = loadRawFolds()
        self.normalized = loadNormFolds()
        self.n_Set = n_Set
        if pca is None:
            self.pca = [D.shape[0]]
        else:
            assert max(pca) <= D.shape[0], f"pca must be smaller than {D.shape[0]}"
            self.pca = pca
        self.print_flag = flag
        self.print_file = "data/Results_PCA5/Results/GMMTied.txt"

    def GMM_EM_Tied(self, X, gmm, psi = 0.01):
        llNew = None
        llOld = None
        G = len(gmm)
        N = X.shape[1]

        while llOld is None or llNew - llOld > 1e-6:
            llOld = llNew
            SJ = numpy.zeros((G, N))
            for g in range(G): #numero componenti
                SJ[g, :] = logpdf_GAU_ND_Opt(X, gmm[g][1], gmm[g][2]) + numpy.log(gmm[g][0])
            SM = scipy.special.logsumexp(SJ, axis = 0)
            llNew = SM.sum() / N
            P = numpy.exp(SJ - SM)
            gmmNew = []
            Z_List = []

            for g in range(G):
                gamma = P[g, :]
                Z = gamma.sum()

                Z_List.append(Z)
                
                F = (vrow(gamma)*X).sum(1)
                S = numpy.dot(X, (vrow(gamma)*X).T)
                w = Z / N
                mu = mcol(F / Z)
                Sigma = S/Z - numpy.dot(mu, mu.T)
                U, s, _ = numpy.linalg.svd(Sigma)
                s[s < psi] = psi
                Sigma = numpy.dot(U, mcol(s) * U.T)

                gmmNew.append((w, mu, Sigma))

            #-----------------------Tied-------------------------#
            gmmTied = []
            sum = numpy.zeros(gmmNew[0][2].shape)

            for g in range(G):
                sum = sum + Z_List[g] * gmm[g][2]

            TiedSigma = sum / X.shape[1]

            for g in range(G):
                gmmTied.append((gmmNew[g][0], gmmNew[g][1], TiedSigma))

            gmm = gmmTied

        return gmm
    
    def GMM_LBG_Tied(self, X, G, alpha = 0.1):
        mu, C = meanAndCovMat(X)
        gmms = []
        
        gmms.append((1.0, mu, C))
        
        gmms = self.GMM_EM_Tied(X, gmms)

        for g in range(G): #G = 2 -> 0, 1
            newList = []
            for element in gmms:
                w = element[0] / 2
                mu = element[1]
                C = element[2]
                U, s, Vh = numpy.linalg.svd(C)
                d = U[:, 0:1] * s[0]**0.5 * alpha
                newList.append((w, mu + d, C))
                newList.append((w, mu - d, C))
            gmms = self.GMM_EM_Tied(X, newList)  

        return gmms 
    
    def kFold(self, folds, n, pca): #KModel è il K relativo al modello
        LLRs = []

        for f in folds:
            DTR = f[0]
            LTR = f[1]
            DTE = f[2]
            LTE = f[3]
            P = dr.PCA_P(DTR, pca)
            DTR = numpy.dot(P.T, DTR)
            DTE = numpy.dot(P.T, DTE)
            DTR0 = DTR[:, LTR == 0] # bad wines
            DTR1 = DTR[:, LTR == 1] # good wines
            gmm0 = self.GMM_LBG_Tied(DTR0, n) #n number of components
            gmm1 = self.GMM_LBG_Tied(DTR1, n) #n number of components
            LLRsRet = GMM_Scores(DTE, gmm0, gmm1)
            LLRs.append(LLRsRet)
        
        LLRs = numpy.hstack(LLRs)

        return LLRs
    
    def calculate_scores(self, DTR, LTR, DTE, n):
        DTR0 = DTR[:, LTR == 0]
        DTR1 = DTR[:, LTR == 1]
        gmm0 = self.GMM_LBG_Tied(DTR0, n)
        gmm1 = self.GMM_LBG_Tied(DTR1, n)
        LLRs = GMM_Scores(DTE, gmm0, gmm1)

        return LLRs
    
    def plot(self, flag: Optional[bool] = True):
        print("Plotting GMM Tied Results...")
        f = open(self.print_file, "r")

        normalized = []
        raw = []

        for line in f:
            elements = line.split("|")
            elements =[elem.strip() for elem in elements]
            if float(float(elements[0]) == 0.5 and elements[4] == "Uncalibrated"):
                if elements[3] == "Raw":
                    raw.append(float(elements[7][9:]))
                if elements[3] == "Normalized":
                    normalized.append(float(elements[7][9:]))

        save_file = f"data/Plots/GMMTied.png"

        plotHistGMM(self.n_Set, raw, normalized, self.type, filename=save_file, flag=flag)
    
    def train(self):
        prior_tilde_set = [0.1, 0.5]

        f = open(self.print_file, "w")

        hyperparameter_list = [(int(math.log2(n)), i) for n in self.n_Set for i in self.pca]

        for n, i in tqdm(hyperparameter_list, desc="Training GMM Tied...", ncols=100):
            Scores = self.kFold(self.raw, n, i)
            for prior_tilde in prior_tilde_set: 
                CalibratedScores, labels = calibrateScores(Scores, self.L, prior_tilde)
                ActDCF, minDCF = me.printDCFs(self.D, self.L, Scores, prior_tilde)
                if self.print_flag:
                    print(f"{prior_tilde} | {self.type} | nComponents = {2**n} | Raw | Uncalibrated | PCA = {i}" + \
                          f" | ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}")
                print(f"{prior_tilde} | {self.type} | nComponents = {2**n} | Raw | Uncalibrated | PCA = {i}" + \
                      f" | ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}", file=f)
                ActDCF, minDCF = me.printDCFs(self.D, labels, CalibratedScores, prior_tilde)
                if self.print_flag:
                    print(f"{prior_tilde} | {self.type} | nComponents = {2**n} | Raw | Calibrated | PCA = {i}" + \
                          f" | ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}")
                print(f"{prior_tilde} | {self.type} | nComponents = {2**n} | Raw | Calibrated | PCA = {i}" + \
                      f" | ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}", file=f)
            
            Scores = self.kFold(self.normalized, n, i)
            for prior_tilde in prior_tilde_set: 
                CalibratedScores, labels = calibrateScores(Scores, self.L, prior_tilde)
                ActDCF, minDCF = me.printDCFs(self.D, self.L, Scores, prior_tilde)
                if self.print_flag:
                    print(f"{prior_tilde} | {self.type} | nComponents = {2**n} | Normalized | Uncalibrated | PCA = {i}" + \
                          f" | ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}")
                print(f"{prior_tilde} | {self.type} | nComponents = {2**n} | Normalized | Uncalibrated | PCA = {i}" + \
                      f" | ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}", file=f)
                ActDCF, minDCF = me.printDCFs(self.D, labels, CalibratedScores, prior_tilde)
                if self.print_flag:
                    print(f"{prior_tilde} | {self.type} | nComponents = {2**n} | Normalized | Calibrated | PCA = {i}" + \
                          f" | ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}")
                print(f"{prior_tilde} | {self.type} | nComponents = {2**n} | Normalized | Calibrated | PCA = {i}" + \
                      f" | ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}", file=f)
                
    def evaluate(self):
        prior_tilde_set = [0.1, 0.5]

        file_path = "data/FinalEvaluation/GMMTied.txt"

        f = open(file_path, "w")

        D_test, L_test = load_test()
        norm_D_test, norm_L_test = load_norm_test()
        D, L = load_train()
        norm_D, L = load_norm_train()

        hyperparameter_list = [(int(math.log2(n)), i) for n in self.n_Set for i in self.pca]

        for n, i in tqdm(hyperparameter_list, desc="Evaluating GMM Tied...", ncols=100):
            D_pca, D_test_pca = apply_PCA(D, D_test, i)
            Scores = self.calculate_scores(D_pca, L, D_test_pca, n)
            for prior_tilde in prior_tilde_set: 
                CalibratedScores, labels = calibrateScores(Scores, L_test, prior_tilde)
                ActDCF, minDCF = me.printDCFs(D_test_pca, L_test, Scores, prior_tilde)
                if self.print_flag:
                    print(f"{prior_tilde} | {self.type} | nComponents = {2**n} | Raw | Uncalibrated | PCA = {i}" + \
                          f" | ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}")
                print(f"{prior_tilde} | {self.type} | nComponents = {2**n} | Raw | Uncalibrated | PCA = {i}" + \
                      f" | ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}", file=f)
                
                ActDCF, minDCF = me.printDCFs(D_test_pca, labels, CalibratedScores, prior_tilde)
                if self.print_flag:
                    print(f"{prior_tilde} | {self.type} | nComponents = {2**n} | Raw | Calibrated | PCA = {i}" + \
                          f" | ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}")
                print(f"{prior_tilde} | {self.type} | nComponents = {2**n} | Raw | Calibrated | PCA = {i}" + \
                      f" | ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}", file=f)
            
            norm_D_pca, norm_D_test_pca = apply_PCA(norm_D, norm_D_test, i)
            Scores = self.calculate_scores(norm_D_pca, L, norm_D_test_pca, n)
            #Still called LLRs in the printDCFs function, but they are scores with no probabilistic interpretation
            #We use the same function for every model
            for prior_tilde in prior_tilde_set: 
                CalibratedScores, labels = calibrateScores(Scores, norm_L_test, prior_tilde)
                ActDCF, minDCF = me.printDCFs(norm_D_test_pca, norm_L_test, Scores, prior_tilde)
                if self.print_flag:
                    print(f"{prior_tilde} | {self.type} | nComponents = {2**n} | Normalized | Uncalibrated | PCA = {i}" + \
                          f" | ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}")
                print(f"{prior_tilde} | {self.type} | nComponents = {2**n} | Normalized | Uncalibrated | PCA = {i}" + \
                      f" | ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}", file=f)
                
                ActDCF, minDCF = me.printDCFs(norm_D_test_pca, labels, CalibratedScores, prior_tilde)
                if self.print_flag:
                    print(f"{prior_tilde} | {self.type} | nComponents = {2**n} | Normalized | Calibrated | PCA = {i}" + \
                          f" | ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}")
                print(f"{prior_tilde} | {self.type} | nComponents = {2**n} | Normalized | Calibrated | PCA = {i}" + \
                      f" | ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}", file=f)