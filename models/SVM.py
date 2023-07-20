from typing import Optional, List
from k_fold_utilities.Raw import loadRawFolds, getShuffledLabels
from k_fold_utilities.Normalized import loadNormFolds
from utils.utils_file import vrow, mcol
import numpy
import scipy
import utils.DimReduction as dr
import utils.ModelEvaluation as me
from tqdm import tqdm
import utils.Plot as plt
from utils.Calibration import calibrateScores
from utils.utils_file import load_test, load_norm_test, load_train, load_norm_train
from utils.DimReduction import apply_PCA

class SVMLinear(object):
    def __init__(self, D, L, K_Set, C_Set,  pca: Optional[List[int]] = None, flag: Optional[bool] = True):
        self.D = D
        self.L = getShuffledLabels()
        self.type = "SVM Linear"
        self.raw = loadRawFolds()
        self.normalized = loadNormFolds()
        self.K_Set = K_Set
        self.C_Set = C_Set
        if pca is None:
            self.pca = [D.shape[0]]
        else:
            assert max(pca) <= D.shape[0], f"pca must be smaller than {D.shape[0]}"
            self.pca = pca
        self.print_flag = flag
        self.print_file = "data/Results_01/SVMLinear.txt"

    def SVMLinear(self, DTR, LTR, DTE, LTE, K, C, prior_t):
        expandedD = numpy.vstack([DTR, K * numpy.ones(DTR.shape[1])])

        Z = numpy.zeros(LTR.shape)
        Z[LTR == 1] = 1
        Z[LTR == 0] = -1
        
        H = numpy.dot(expandedD.T, expandedD)
        H = mcol(Z) * vrow(Z) * H

        def JDual(alpha):
            Ha = numpy.dot(H, mcol(alpha))
            aHa = numpy.dot(vrow(alpha), Ha)
            a1 = alpha.sum()

            return -0.5 * aHa.ravel() + a1, -Ha.ravel() + numpy.ones(alpha.size)

        def LDual(alpha):
            loss, grad = JDual(alpha)
            return -loss, -grad

        ##
            
        boundaries = numpy.empty(LTR.shape, dtype = 'f,f')
        Pi_T_Emp = LTR[LTR == 1].size / LTR.size
        Pi_F_Emp = LTR[LTR == 0].size / LTR.size
        Ct = C * prior_t / Pi_T_Emp
        Cf = C * (1 - prior_t) / Pi_F_Emp
        boundaries[LTR == 0] = (0, Cf)
        boundaries[LTR == 1] = (0, Ct)
        
        alphaStar, x, y = scipy.optimize.fmin_l_bfgs_b(LDual,
                                                    numpy.zeros(DTR.shape[1]),
                                                    bounds = boundaries,
                                                    factr = 1.0,
                                                    maxiter=5000,
                                                    maxfun=100000
                                                    )
        
        wStar = numpy.dot(expandedD, mcol(alphaStar) * mcol(Z))

        expandedDTE = numpy.vstack([DTE, K * numpy.ones(DTE.shape[1])])
        score = numpy.dot(wStar.T, expandedDTE)
        Predictions = score > 0

        return score[0]
    
    def kFold(self, folds, KModel, C, prior_t, pca): #KModel è il K relativo al modello

        LLRs = []

        for f in folds:
            DTR = f[0]
            LTR = f[1]
            DTE = f[2]
            LTE = f[3]
            P = dr.PCA_P(DTR, pca)
            DTR = numpy.dot(P.T, DTR)
            DTE = numpy.dot(P.T, DTE)
            LLRsRet = self.SVMLinear(DTR, LTR, DTE, LTE, KModel, C, prior_t)
            LLRs.append(LLRsRet)
        
        LLRs = numpy.hstack(LLRs)

        return LLRs
    
    def plot(self, flag: Optional[bool] = True):
        print("PLotting SVM Linears results...")
        f = open(self.print_file, "r")

        normalized=[]
        raw=[]

        #   0.1 | 0.1 | SVM Linear | K = 0.0 | C = 0.01 | Raw | Uncalibrated | PCA = 10 | ActDCF =1.000 | MinDCF =0.993

        #(prior , MinDCF , K , C)
        for line in f:
            elements = line.split("|")
            elements =[elem.strip() for elem in elements]
            if(float(elements[0]) == 0.1):
                if(elements[5]=="Normalized" and elements[6]=="Uncalibrated"):
                    normalized.append(( float(elements[1]), float(elements[9][8:]), float(elements[3][4:]), float(elements[4][4:]) ))

                elif(elements[5]=="Raw" and elements[6]=="Uncalibrated"):
                    raw.append(( float(elements[1]), float(elements[9][8:]), float(elements[3][4:]), float(elements[4][4:]) ))

        bestNorm05 = min(raw, key=lambda x: x[1])
        best_K_Norm05 = bestNorm05[2]
        print(f"Best K for 0.5 Normalized: {best_K_Norm05}")
        normalized05 = numpy.array([x[1] for x in normalized if x[2] == best_K_Norm05 and x[0] == 0.5])   

        bestNorm01 = min(raw, key=lambda x: x[1])
        best_K_Norm01 = bestNorm01[2]
        print(f"Best K for 0.1 Normalized: {best_K_Norm01}")
        normalized01 = numpy.array([x[1] for x in normalized if x[2] == best_K_Norm01 and x[0] == 0.1]) 
    
        bestRaw05 = min(raw, key=lambda x: x[1])
        best_K_Raw05 = bestRaw05[2]
        print(f"Best K for 0.5 Raw: {best_K_Raw05}")
        raw05 = numpy.array([x[1] for x in raw if x[2] == best_K_Raw05 and x[0] == 0.5]) 

        bestRaw01 = min(raw, key=lambda x: x[1])
        best_K_Raw01 = bestRaw01[2]
        print(f"Best K for 0.1 Raw: {best_K_Raw01}")
        raw01 = numpy.array([x[1] for x in raw if x[2] == best_K_Raw01 and x[0] == 0.1]) 
 

        norm_plot_file = f"data/Plots/SVMLinear_{best_K_Raw05}_Norm.png"
        raw_plot_file = f"data/Plots/SVMLinear_{best_K_Raw05}_Raw.png"

        plt.plotTwoDCFs(self.C_Set, normalized05, normalized01, "C", "Normalized", norm_plot_file, flag=flag)
        plt.plotTwoDCFs(self.C_Set, raw05, raw01, "C", "Raw", raw_plot_file, flag=flag)
 
    def train(self, prior_t): #K relativo al modello, non k_fold
        prior_tilde_set = [0.1, 0.5, 0.9]

        f = open(self.print_file, "w")

        hyperparameter_list = [(K, C, i) for K in self.K_Set for C in self.C_Set for i in self.pca]
        
        for K, C, i in tqdm(hyperparameter_list, "Training SVM Linear...", ncols=100):
            Scores = self.kFold(self.raw, K, C, prior_t, i)
            #Still called LLRs in the printDCFs function, but they are scores with no probabilistic interpretation
            #We use the same function for every model
            CalibratedScores, labels = calibrateScores(Scores, self.L, prior_t)
            for prior_tilde in prior_tilde_set:
                ActDCF, minDCF = me.printDCFs(self.D, self.L, Scores, prior_tilde)
                if self.print_flag:
                    print(f"{prior_t} | {prior_tilde} | {self.type} | K = {K} | C = {C} | Raw | Uncalibrated | PCA = {i}" + \
                          f" | ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}")
                print(f"{prior_t} | {prior_tilde} | {self.type} | K = {K} | C = {C} | Raw | Uncalibrated | PCA = {i}" + \
                      f" | ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}", file=f)
                ActDCF, minDCF = me.printDCFs(self.D, labels, CalibratedScores, prior_tilde)
                if self.print_flag:
                    print(f"{prior_t} | {prior_tilde} | {self.type} | K = {K} | C = {C} | Raw | Calibrated | PCA = {i}" + \
                          f" | ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}")
                print(f"{prior_t} | {prior_tilde} | {self.type} | K = {K} | C = {C} | Raw | Calibrated | PCA = {i}" + \
                      f" | ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}", file=f)

            Scores = self.kFold(self.normalized, K, C, prior_t, i)
            CalibratedScores, labels = calibrateScores(Scores, self.L, prior_t)
            for prior_tilde in prior_tilde_set:
                ActDCF, minDCF = me.printDCFs(self.D, self.L, Scores, prior_tilde) 
                if self.print_flag:
                    print(f"{prior_t} | {prior_tilde} | {self.type} | K = {K} | C = {C} | Normalized | Uncalibrated | PCA = {i}" + \
                            f" | ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}")
                print(f"{prior_t} | {prior_tilde} | {self.type} | K = {K} | C = {C} | Normalized | Uncalibrated | PCA = {i}" + \
                      f" | ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}", file=f)
                ActDCF, minDCF = me.printDCFs(self.D, labels, CalibratedScores, prior_tilde) 
                if self.print_flag:
                    print(f"{prior_t} | {prior_tilde} | {self.type} | K = {K} | C = {C} | Normalized | Calibrated | PCA = {i}" + \
                          f" | ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}")
                print(f"{prior_t} | {prior_tilde} | {self.type} | K = {K} | C = {C} | Normalized | Calibrated | PCA = {i}" + \
                      f" | ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}", file=f)
                
    def evaluate(self, prior_t): #K relativo al modello, non k_fold
        prior_tilde_set = [0.1, 0.5]

        string = str(prior_t).replace(".","")

        file_path = f"data/FinalEvaluation/SVMLinear_{string}.txt"

        f = open(file_path, "w")

        D_test, L_test = load_test()
        norm_D_test, norm_L_test = load_norm_test()
        D, L = load_train()
        norm_D, L = load_norm_train()

        hyperparameter_list = [(K, C, i) for K in self.K_Set for C in self.C_Set for i in self.pca]
        
        for K, C, i in tqdm(hyperparameter_list, "Evaluating SVM Linear...", ncols=100):
            D_pca, D_test_pca = apply_PCA(D, D_test, i)
            Scores = self.SVMLinear(D_pca, L, D_test_pca, L_test, K, C, prior_t)
            #Still called LLRs in the printDCFs function, but they are scores with no probabilistic interpretation
            #We use the same function for every model
            CalibratedScores, labels = calibrateScores(Scores, L_test, prior_t)
            for prior_tilde in prior_tilde_set:
                ActDCF, minDCF = me.printDCFs(D_test_pca, L_test, Scores, prior_tilde)
                if self.print_flag:
                    print(f"{prior_t} | {prior_tilde} | {self.type} | K = {K} | C = {C} | Raw | Uncalibrated | PCA = {i}" + \
                          f" | ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}")
                print(f"{prior_t} | {prior_tilde} | {self.type} | K = {K} | C = {C} | Raw | Uncalibrated | PCA = {i}" + \
                      f" | ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}", file=f)
                
                ActDCF, minDCF = me.printDCFs(D_test_pca, labels, CalibratedScores, prior_tilde)
                if self.print_flag:
                    print(f"{prior_t} | {prior_tilde} | {self.type} | K = {K} | C = {C} | Raw | Calibrated | PCA = {i}" + \
                          f" | ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}")
                print(f"{prior_t} | {prior_tilde} | {self.type} | K = {K} | C = {C} | Raw | Calibrated | PCA = {i}" + \
                      f" | ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}", file=f)
                
            norm_D_pca, norm_D_test_pca = apply_PCA(norm_D, norm_D_test, i)
            Scores = self.SVMLinear(norm_D_pca, L, norm_D_test_pca, norm_L_test, K, C, prior_t)

            CalibratedScores, labels = calibrateScores(Scores, norm_L_test, prior_t)

            for prior_tilde in prior_tilde_set:
                ActDCF, minDCF = me.printDCFs(norm_D_test_pca, norm_L_test, Scores, prior_tilde) 
                if self.print_flag:
                    print(f"{prior_t} | {prior_tilde} | {self.type} | K = {K} | C = {C} | Normalized | Uncalibrated | PCA = {i}" + \
                            f" | ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}")
                print(f"{prior_t} | {prior_tilde} | {self.type} | K = {K} | C = {C} | Normalized | Uncalibrated | PCA = {i}" + \
                      f" | ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}", file=f)
                
                ActDCF, minDCF = me.printDCFs(norm_D_test_pca, labels, CalibratedScores, prior_tilde) 
                if self.print_flag:
                    print(f"{prior_t} | {prior_tilde} | {self.type} | K = {K} | C = {C} | Normalized | Calibrated | PCA = {i}" + \
                          f" | ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}")
                print(f"{prior_t} | {prior_tilde} | {self.type} | K = {K} | C = {C} | Normalized | Calibrated | PCA = {i}" + \
                      f" | ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}", file=f)
                

class SVMPoly(object):
    def __init__(self, D, L, K_Set, C_Set, d_Set, c_Set, pca: Optional[List[int]] = None, flag: Optional[bool] = True):
        self.D = D
        self.L = L
        self.type = "SVM Poly"
        self.raw = loadRawFolds()
        self.normalized = loadNormFolds()
        self.K_Set = K_Set
        self.C_Set = C_Set
        self.d_Set = d_Set
        self.c_Set = c_Set
        if pca is None:
            self.pca = [D.shape[0]]
        else:
            assert max(pca) <= D.shape[0], f"pca must be smaller than {D.shape[0]}"
            self.pca = pca
        self.print_flag = flag
        self.print_file = "data/SVMPoly_4dim_01.txt"

    def SVMPoly(self, DTR, LTR, DTE, LTE, K, C, d, c, prior_t):
        Z = numpy.zeros(LTR.shape)
        Z[LTR == 1] = 1
        Z[LTR == 0] = -1
        
        epsilon = K**2

        product = numpy.dot(DTR.T, DTR)
        Kernel = (product + c)**d + epsilon
        H = mcol(Z) * vrow(Z) * Kernel

        def JDual(alpha):
            Ha = numpy.dot(H, mcol(alpha))
            aHa = numpy.dot(vrow(alpha), Ha)
            a1 = alpha.sum()

            return -0.5 * aHa.ravel() + a1, -Ha.ravel() + numpy.ones(alpha.size)

        def LDual(alpha):
            loss, grad = JDual(alpha)
            return -loss, -grad

        ##
        
        boundaries = numpy.empty(LTR.shape, dtype = 'f,f')
        Pi_T_Emp = (LTR == 1).size / LTR.size
        Pi_F_Emp = (LTR == 0).size / LTR.size

        Ct = C * prior_t / Pi_T_Emp
        Cf = C * (1 - prior_t) / Pi_F_Emp
        boundaries[LTR == 0] = (0, Cf)
        boundaries[LTR == 1] = (0, Ct)
        alphaStar, x, y = scipy.optimize.fmin_l_bfgs_b(LDual,
                                                    numpy.zeros(DTR.shape[1]),
                                                    bounds = boundaries,
                                                    factr = 1.0,
                                                    maxiter=5000,
                                                    maxfun=100000
                                                    )
        
        scores = []
        for x_t in DTE.T:
            score = 0
            for i in range(DTR.shape[1]):
                Kernel = (numpy.dot(DTR.T[i].T, x_t) + c)**d + epsilon
                score += alphaStar[i] * Z[i] * Kernel
            scores.append(score)
        
        scores = numpy.hstack(scores)
        
        Predictions = scores > 0
        Predictions = numpy.hstack(Predictions)

        return scores
    
    def kFold(self, folds, KModel, C, d, c, prior_t, pca): #KModel è il K relativo al modello

        LLRs = []

        for f in folds:
            DTR = f[0]
            LTR = f[1]
            DTE = f[2]
            LTE = f[3]
            P = dr.PCA_P(DTR, pca)
            DTR = numpy.dot(P.T, DTR)
            DTE = numpy.dot(P.T, DTE)
            LLRsRet = self.SVMPoly(DTR, LTR, DTE, LTE, KModel, C, d, c, prior_t)
            LLRs.append(LLRsRet)
        
        LLRs = numpy.hstack(LLRs)

        return LLRs
    
    def plot(self, flag: Optional[bool] = True):
        print("Plotting SVM Poly results...")
        f = open(self.print_file, "r")
        i_MinDCF = []
        lines = []

        #0.1 | 0.1 | SVM Poly | K = 0.0 | C = 0.01 | d = 2.0 | c = 0.0 | Raw | Uncalibrated | PCA = 10 | ActDCF =1.737 | MinDCF =0.997

        for i, line in enumerate(f):
            elements = line.split("|")
            elements =[elem.strip() for elem in elements]
            lines.append(elements)
            MinDCF = elements[11][8:]
            i_MinDCF.append((i, float(elements[0]), MinDCF)) #(indice, prior, mindcf)
        
        i_MinDCF05 = filter(lambda x: x[1] == 0.1, i_MinDCF)
        MinDCF = min(i_MinDCF05, key = lambda x: x[2])
        #print(MinDCF)
        index = MinDCF[0]
        #print(lines[index])

        Best_K = lines[index][3]
        Best_d = lines[index][5]
        Best_c = lines[index][6]
        #Best_c = "c = 1.0"
        raw05 = []
        raw01 = []
        normalized05 = []
        normalized01 = []

        for line in lines:
            DataType = line[7]
            Cal = line[8]
            prior_t = float(line[0])
            pi_tilde = float(line[1])
            K = line[3]
            d = line[5]
            c = line[6]
            minDCF = float(line[11][8:])

            if (prior_t == 0.1 and Cal == "Uncalibrated"):
                if (K == Best_K and d == Best_d and c == Best_c):
                    if(DataType == "Raw"):
                        if(pi_tilde == 0.5):
                            raw05.append(minDCF)
                        if(pi_tilde == 0.1):
                            raw01.append(minDCF)

                    if(DataType == "Normalized"):
                        if(pi_tilde == 0.5):
                            normalized05.append(minDCF)
                        if(pi_tilde == 0.1):
                            normalized01.append(minDCF)

        norm_plot_file = f"data/Plots/SVMPoly_{Best_K}_{Best_d}_{Best_c}_Norm_01.png"
        raw_plot_file = f"data/Plots/SVMPoly_{Best_K}_{Best_d}_{Best_c}_Raw_01.png"
                            
        plt.plotTwoDCFs(self.C_Set, raw05, raw01, "C", "Raw", raw_plot_file, flag=flag)
        plt.plotTwoDCFs(self.C_Set, normalized05, normalized01, "C", "Normalized", norm_plot_file, flag=flag)
        
    def train(self, prior_t):
        prior_tilde_set = [0.1, 0.5, 0.9]

        f = open(self.print_file, "w")

        hyperparameter_list = [(K, C, d, c, i) for K in self.K_Set for C in self.C_Set for d in self.d_Set for c in self.c_Set for i in self.pca]

        for K, C, d, c, i in tqdm(hyperparameter_list, desc="Training SVM Poly...", ncols=100):
            Scores = self.kFold(self.raw, K, C, d, c, prior_t, i)
            CalibratedScores, labels = calibrateScores(Scores, self.L, prior_t)
            for prior_tilde in prior_tilde_set: 
                ActDCF, minDCF = me.printDCFs(self.D, self.L, Scores, prior_tilde)
                if self.print_flag:
                    print(f"{prior_t} | {prior_tilde} | {self.type} | K = {K} | C = {C} | d = {d} | c = {c} | Raw | Uncalibrated | PCA = {i}" + \
                          f" | ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}")
                print(f"{prior_t} | {prior_tilde} | {self.type} | K = {K} | C = {C} | d = {d} | c = {c} | Raw | Uncalibrated | PCA = {i}" + \
                      f" | ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}", file=f)
                ActDCF, minDCF = me.printDCFs(self.D, labels, CalibratedScores, prior_tilde) 
                if self.print_flag:
                    print(f"{prior_t} | {prior_tilde} | {self.type} | K = {K} | C = {C} | d = {d} | c = {c} | Raw | Calibrated | PCA = {i}" + \
                          f" | ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}")
                print(f"{prior_t} | {prior_tilde} | {self.type} | K = {K} | C = {C} | d = {d} | c = {c} | Raw | Calibrated | PCA = {i}" + \
                      f" | ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}", file=f)
            
            Scores = self.kFold(self.normalized, K, C, d, c, prior_t, i)
            CalibratedScores, labels = calibrateScores(Scores, self.L, prior_t)
            for prior_tilde in prior_tilde_set: 
                ActDCF, minDCF = me.printDCFs(self.D, self.L, Scores, prior_tilde) 
                if self.print_flag:
                    print(f"{prior_t} | {prior_tilde} | {self.type} | K = {K} | C = {C} | d = {d} | c = {c} | Normalized | Uncalibrated | PCA = {i}" + \
                          f" | ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}")
                print(f"{prior_t} | {prior_tilde} | {self.type} | K = {K} | C = {C} | d = {d} | c = {c} | Normalized | Uncalibrated | PCA = {i}" + \
                      f" | ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}", file=f)
                ActDCF, minDCF = me.printDCFs(self.D, labels, CalibratedScores, prior_tilde) 
                if self.print_flag:
                    print(f"{prior_t} | {prior_tilde} | {self.type} | K = {K} | C = {C} | d = {d} | c = {c} | Normalized | Calibrated | PCA = {i}" + \
                          f" | ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}")
                print(f"{prior_t} | {prior_tilde} | {self.type} | K = {K} | C = {C} | d = {d} | c = {c} | Normalized | Calibrated | PCA = {i}" + \
                      f" | ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}", file=f)
                
    def evaluate(self, prior_t):
        prior_tilde_set = [0.1, 0.5]

        string = str(prior_t).replace(".","")

        file_path = f"data/FinalEvaluation/SVMPoly_{string}.txt"

        f = open(file_path, "w")

        D_test, L_test = load_test()
        norm_D_test, norm_L_test = load_norm_test()
        D, L = load_train()
        norm_D, L = load_norm_train()

        hyperparameter_list = [(K, C, d, c, i) for K in self.K_Set for C in self.C_Set for d in self.d_Set for c in self.c_Set for i in self.pca]

        for K, C, d, c, i in tqdm(hyperparameter_list, desc="Evaluating SVM Poly...", ncols=100):
            D_pca, D_test_pca = apply_PCA(D, D_test, i)
            Scores = self.SVMPoly(D_pca, L, D_test_pca, L_test, K, C, d, c, prior_t)
            CalibratedScores, labels = calibrateScores(Scores, L_test, prior_t)
            for prior_tilde in prior_tilde_set: 
                ActDCF, minDCF = me.printDCFs(D_test_pca, L_test, Scores, prior_tilde)
                if self.print_flag:
                    print(f"{prior_t} | {prior_tilde} | {self.type} | K = {K} | C = {C} | d = {d} | c = {c} | Raw | Uncalibrated | PCA = {i}" + \
                          f" | ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}")
                print(f"{prior_t} | {prior_tilde} | {self.type} | K = {K} | C = {C} | d = {d} | c = {c} | Raw | Uncalibrated | PCA = {i}" + \
                      f" | ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}", file=f)
                
                ActDCF, minDCF = me.printDCFs(D_test_pca, labels, CalibratedScores, prior_tilde)
                if self.print_flag:
                    print(f"{prior_t} | {prior_tilde} | {self.type} | K = {K} | C = {C} | d = {d} | c = {c} | Raw | Calibrated | PCA = {i}" + \
                          f" | ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}")
                print(f"{prior_t} | {prior_tilde} | {self.type} | K = {K} | C = {C} | d = {d} | c = {c} | Raw | Calibrated | PCA = {i}" + \
                      f" | ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}", file=f)
            
            norm_D_pca, norm_D_test_pca = apply_PCA(norm_D, norm_D_test, i)
            Scores = self.SVMPoly(norm_D_pca, L, norm_D_test_pca, norm_L_test, K, C, d, c, prior_t)

            CalibratedScores, labels = calibrateScores(Scores, norm_L_test, prior_t)
            
            for prior_tilde in prior_tilde_set: 
                ActDCF, minDCF = me.printDCFs(norm_D_test_pca, norm_L_test, Scores, prior_tilde) 
                if self.print_flag:
                    print(f"{prior_t} | {prior_tilde} | {self.type} | K = {K} | C = {C} | d = {d} | c = {c} | Normalized | Uncalibrated | PCA = {i}" + \
                          f" | ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}")
                print(f"{prior_t} | {prior_tilde} | {self.type} | K = {K} | C = {C} | d = {d} | c = {c} | Normalized | Uncalibrated | PCA = {i}" + \
                      f" | ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}", file=f)
                
                ActDCF, minDCF = me.printDCFs(norm_D_test_pca, labels, CalibratedScores, prior_tilde)  
                if self.print_flag:
                    print(f"{prior_t} | {prior_tilde} | {self.type} | K = {K} | C = {C} | d = {d} | c = {c} | Normalized | Calibrated | PCA = {i}" + \
                          f" | ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}")
                print(f"{prior_t} | {prior_tilde} | {self.type} | K = {K} | C = {C} | d = {d} | c = {c} | Normalized | Calibrated | PCA = {i}" + \
                      f" | ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}", file=f)



class SVMRBF(object):
    def __init__(self, D, L, K_Set, C_Set, gamma_Set, pca: Optional[List[int]] = None, flag: Optional[bool] = True):
        self.D = D
        self.L = L
        self.type = "SVM RBF"
        self.raw = loadRawFolds()
        self.normalized = loadNormFolds()
        self.K_Set = K_Set
        self.C_Set = C_Set
        self.gamma_Set = gamma_Set
        if pca is None:
            self.pca = [D.shape[0]]
        else:
            assert max(pca) <= D.shape[0], f"pca must be smaller than {D.shape[0]}"
            self.pca = pca
        self.print_flag = flag
        self.print_file = "data/Results_01/SVMRBF.txt"

    def SVM_RBF(self, DTR, LTR, DTE, LTE, K, C, gamma, prior_t):
        Z = numpy.zeros(LTR.shape)
        Z[LTR == 1] = 1
        Z[LTR == 0] = -1
        
        epsilon = K**2

        Dist = numpy.zeros([DTR.shape[1], DTR.shape[1]])

        for i in range(DTR.shape[1]):
            for j in range(DTR.shape[1]):
                xi = DTR[:, i]
                xj = DTR[:, j]
                Dist[i, j] = numpy.linalg.norm(xi - xj)**2

        Kernel = numpy.exp(- gamma * Dist) + epsilon
        H = mcol(Z) * vrow(Z) * Kernel

        def JDual(alpha):
            Ha = numpy.dot(H, mcol(alpha))
            aHa = numpy.dot(vrow(alpha), Ha)
            a1 = alpha.sum()

            return -0.5 * aHa.ravel() + a1, -Ha.ravel() + numpy.ones(alpha.size)

        def LDual(alpha):
            loss, grad = JDual(alpha)
            return -loss, -grad

        ##
        
        boundaries = numpy.empty(LTR.shape, dtype = 'f,f')
        Pi_T_Emp = (LTR == 1).size / LTR.size
        Pi_F_Emp = (LTR == 0).size / LTR.size

        Ct = C * prior_t / Pi_T_Emp
        Cf = C * (1 - prior_t) / Pi_F_Emp
        boundaries[LTR == 0] = (0, Cf)
        boundaries[LTR == 1] = (0, Ct)
        alphaStar, x, y = scipy.optimize.fmin_l_bfgs_b(LDual,
                                                    numpy.zeros(DTR.shape[1]),
                                                    bounds=boundaries,
                                                    factr = 1.0,
                                                    maxiter=5000,
                                                    maxfun=100000
                                                    )
        
        scores = []
        for x_t in DTE.T:
            score = 0
            for i in range(DTR.shape[1]):
                Dist = numpy.linalg.norm(DTR[:, i] - x_t)**2
                Kernel = numpy.exp(- gamma * Dist) + epsilon
                score += alphaStar[i] * Z[i] * Kernel
            scores.append(score)
        
        scores = numpy.hstack(scores)
        
        Predictions = scores > 0
        Predictions = numpy.hstack(Predictions)

        return scores
    
    def kFold(self, folds, KModel, C, gamma, prior_t, pca): #KModel è il K relativo al modello
        LLRs = []

        for f in folds:
            DTR = f[0]
            LTR = f[1]
            DTE = f[2]
            LTE = f[3]
            P = dr.PCA_P(DTR, pca)
            DTR = numpy.dot(P.T, DTR)
            DTE = numpy.dot(P.T, DTE)
            LLRsRet = self.SVM_RBF(DTR, LTR, DTE, LTE, KModel, C, gamma, prior_t)
            LLRs.append(LLRsRet)
        
        LLRs = numpy.hstack(LLRs)

        return LLRs

    def plot(self, flag: Optional[bool] = True):
        print("Plotting SVM RBF results...")
        f = open(self.print_file, "r")

        i_MinDCF = []
        lines = []

        #0.1 | 0.1 | SVM RBF | K = 0.0 | C = 0.01 | gamma = 0.001 | Raw | Uncalibrated | PCA = 10 | ActDCF =1.000 | MinDCF =1.000

        for i, line in enumerate(f):
            elements = line.split("|")
            elements =[elem.strip() for elem in elements]
            lines.append(elements)
            if (elements[7] == "Uncalibrated"):
                MinDCF = elements[10][8:]
                i_MinDCF.append((i, float(elements[0]), MinDCF)) #(indice, prior, mindcf)
        
        i_MinDCF05 = filter(lambda x: x[1] == 0.5, i_MinDCF)
        MinDCF = min(i_MinDCF05, key = lambda x: x[2])
        print(MinDCF)
        index = MinDCF[0]
        print(lines[index])

        Best_K = lines[index][3]
        Best_C = lines[index][4]
        raw_gamma_0 = []
        raw_gamma_1 = []
        raw_gamma_2 = []
        normalized_gamma_0 = []
        normalized_gamma_1 = []
        normalized_gamma_2 = []


        for line in lines:
            DataType = line[6]
            Cal = line[7]
            PCA = line[8]
            prior_t = float(line[0])
            pi_tilde = float(line[1])
            K = line[3]
            C = line[4]
            gamma = float(line[5][8:])
            minDCF = float(line[10][8:])

            if (prior_t == 0.5 and Cal == "Uncalibrated" and pi_tilde == 0.5 and PCA == "PCA = 6"):
                if (K == Best_K):
                    if(DataType == "Raw"):
                        if(gamma == self.gamma_Set[0]):
                            raw_gamma_0.append(minDCF)
                        if(gamma == self.gamma_Set[1]):
                            raw_gamma_1.append(minDCF)
                        if(gamma == self.gamma_Set[2]):
                            raw_gamma_2.append(minDCF)

                    if(DataType == "Normalized"):
                        if(gamma == self.gamma_Set[0]):
                            normalized_gamma_0.append(minDCF)
                        if(gamma == self.gamma_Set[1]):
                            normalized_gamma_1.append(minDCF)
                        if(gamma == self.gamma_Set[2]):
                            normalized_gamma_2.append(minDCF)

        norm_plot_file = f"data/Plots/SVMRBF_{Best_K}_Norm.png"
        raw_plot_file = f"data/Plots/SVMRBF_{Best_K}_Raw.png"

        plt.plotThreeDCFsRBF(self.C_Set, raw_gamma_0, raw_gamma_1, raw_gamma_2, "C", "Raw", raw_plot_file, flag=flag)
        plt.plotThreeDCFsRBF(self.C_Set, normalized_gamma_0, normalized_gamma_1, normalized_gamma_2, "C", "Normalized", norm_plot_file, flag=flag)
    
    def train(self, prior_t):
        prior_tilde_set = [0.1, 0.5]

        f = open(self.print_file, "w")

        hyperparameter_list = [(K, C, gamma, i) for K in self.K_Set for C in self.C_Set for gamma in self.gamma_Set for i in self.pca]

        for K, C, gamma, i in tqdm(hyperparameter_list, desc="Training SVM RBF...", ncols=100):
            Scores = self.kFold(self.raw, K, C, gamma, prior_t, i)
            CalibratedScores, labels = calibrateScores(Scores, self.L, prior_t)
            for prior_tilde in prior_tilde_set: 
                ActDCF, minDCF = me.printDCFs(self.D, self.L, Scores, prior_tilde) 
                if self.print_flag:
                    print(f"{prior_t} | {prior_tilde} | {self.type} | K = {K} | C = {C} | gamma = {gamma} | Raw | Uncalibrated | PCA = {i}" + \
                          f" | ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}")
                print(f"{prior_t} | {prior_tilde} | {self.type} | K = {K} | C = {C} | gamma = {gamma} | Raw | Uncalibrated | PCA = {i}" + \
                      f" | ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}", file=f)
                ActDCF, minDCF = me.printDCFs(self.D, labels, CalibratedScores, prior_tilde) 
                if self.print_flag:
                    print(f"{prior_t} | {prior_tilde} | {self.type} | K = {K} | C = {C} | gamma = {gamma} | Raw | Calibrated | PCA = {i}" + \
                          f" | ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}")
                print(f"{prior_t} | {prior_tilde} | {self.type} | K = {K} | C = {C} | gamma = {gamma} | Raw | Calibrated | PCA = {i}" + \
                      f" | ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}", file=f)


            Scores = self.kFold(self.normalized, K, C, gamma, prior_t, i)
            CalibratedScores, labels = calibrateScores(Scores, self.L, prior_t)
            for prior_tilde in prior_tilde_set: 
                ActDCF, minDCF = me.printDCFs(self.D, self.L, Scores, prior_tilde) 
                if self.print_flag:
                    print(f"{prior_t} | {prior_tilde} | {self.type} | K = {K} | C = {C} | gamma = {gamma} | Normalized | Uncalibrated | PCA = {i}" + \
                          f" | ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}")
                print(f"{prior_t} | {prior_tilde} | {self.type} | K = {K} | C = {C} | gamma = {gamma} | Normalized | Uncalibrated | PCA = {i}" + \
                      f" | ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}", file=f)
                ActDCF, minDCF = me.printDCFs(self.D, labels, CalibratedScores, prior_tilde) 
                if self.print_flag:
                    print(f"{prior_t} | {prior_tilde} | {self.type} | K = {K} | C = {C} | gamma = {gamma} | Normalized | Calibrated | PCA = {i}" + \
                          f" | ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}")
                print(f"{prior_t} | {prior_tilde} | {self.type} | K = {K} | C = {C} | gamma = {gamma} | Normalized | Calibrated | PCA = {i}" + \
                      f" | ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}", file=f)

    def evaluate(self, prior_t):
        prior_tilde_set = [0.1, 0.5]

        string = str(prior_t).replace(".","")

        file_path = f"data/FinalEvaluation/SVMRBF_{string}.txt"

        f = open(file_path, "w")

        D_test, L_test = load_test()
        norm_D_test, norm_L_test = load_norm_test()
        D, L = load_train()
        norm_D, L = load_norm_train()

        hyperparameter_list = [(K, C, gamma, i) for K in self.K_Set for C in self.C_Set for gamma in self.gamma_Set for i in self.pca]

        for K, C, gamma, i in tqdm(hyperparameter_list, desc="Evaluating SVM RBF...", ncols=100):
            D_pca, D_test_pca = apply_PCA(D, D_test, i)
            Scores = self.SVM_RBF(D_pca, L, D_test_pca, L_test, K, C, gamma, prior_t)
            CalibratedScores, labels = calibrateScores(Scores, L_test, prior_t)
            for prior_tilde in prior_tilde_set: 
                ActDCF, minDCF = me.printDCFs(D_test_pca, L_test, Scores, prior_tilde)
                if self.print_flag:
                    print(f"{prior_t} | {prior_tilde} | {self.type} | K = {K} | C = {C} | gamma = {gamma} | Raw | Uncalibrated | PCA = {i}" + \
                          f" | ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}")
                print(f"{prior_t} | {prior_tilde} | {self.type} | K = {K} | C = {C} | gamma = {gamma} | Raw | Uncalibrated | PCA = {i}" + \
                      f" | ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}", file=f)
                
                ActDCF, minDCF = me.printDCFs(D_test_pca, labels, CalibratedScores, prior_tilde)
                if self.print_flag:
                    print(f"{prior_t} | {prior_tilde} | {self.type} | K = {K} | C = {C} | gamma = {gamma} | Raw | Calibrated | PCA = {i}" + \
                          f" | ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}")
                print(f"{prior_t} | {prior_tilde} | {self.type} | K = {K} | C = {C} | gamma = {gamma} | Raw | Calibrated | PCA = {i}" + \
                      f" | ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}", file=f)


            norm_D_pca, norm_D_test_pca = apply_PCA(norm_D, norm_D_test, i)
            Scores = self.SVM_RBF(norm_D_pca, L, norm_D_test_pca, norm_L_test, K, C, gamma, prior_t)

            CalibratedScores, labels = calibrateScores(Scores, norm_L_test, prior_t)

            for prior_tilde in prior_tilde_set: 
                ActDCF, minDCF = me.printDCFs(norm_D_test_pca, norm_L_test, Scores, prior_tilde)
                if self.print_flag:
                    print(f"{prior_t} | {prior_tilde} | {self.type} | K = {K} | C = {C} | gamma = {gamma} | Normalized | Uncalibrated | PCA = {i}" + \
                          f" | ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}")
                print(f"{prior_t} | {prior_tilde} | {self.type} | K = {K} | C = {C} | gamma = {gamma} | Normalized | Uncalibrated | PCA = {i}" + \
                      f" | ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}", file=f)
                
                ActDCF, minDCF = me.printDCFs(norm_D_test_pca, labels, CalibratedScores, prior_tilde)
                if self.print_flag:
                    print(f"{prior_t} | {prior_tilde} | {self.type} | K = {K} | C = {C} | gamma = {gamma} | Normalized | Calibrated | PCA = {i}" + \
                          f" | ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}")
                print(f"{prior_t} | {prior_tilde} | {self.type} | K = {K} | C = {C} | gamma = {gamma} | Normalized | Calibrated | PCA = {i}" + \
                      f" | ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}", file=f)