from typing import Optional, List
from k_fold_utilities.Raw import loadRawFolds
from k_fold_utilities.Normalized import loadNormFolds
import numpy
import utils.DimReduction as dr
from utils.utils_file import mcol, vrow
import scipy.optimize
import utils.ModelEvaluation as me
import utils.Plot as plt
from tqdm import tqdm
from utils.utils_file import load_test, load_norm_test, load_train, load_norm_train
from utils.DimReduction import apply_PCA

def logreg_obj(v, DTR, LTR, l, prior):
    w, b = mcol(v[0:-1]), v[-1]

    LTR0 = LTR[LTR == 0]
    LTR1 = LTR[LTR == 1]

    Z0 = LTR0 * 2.0 - 1.0
    Z1 = LTR1 * 2.0 - 1.0

    S0 = numpy.dot(w.T, DTR[:, LTR == 0]) + b
    S1 = numpy.dot(w.T, DTR[:, LTR == 1]) + b
    
    NF = len(LTR0)
    NT = len(LTR1)
    
    cxeF = numpy.logaddexp(0, -Z0*S0).sum() * (1 - prior) / NF
    cxeT = numpy.logaddexp(0, -Z1*S1).sum() * prior / NT

    return l/2 * numpy.linalg.norm(w)**2 + cxeT + cxeF

def logreg_obj_w_b(v, score, labels, l):
    w, b = mcol(v[0:-1]), v[-1]
    Z = labels * 2.0 - 1.0
    score = vrow(score)
    S = numpy.dot(w.T, score) + b
    cxe = numpy.logaddexp(0, -Z*S).mean()
    return l/2 * numpy.linalg.norm(w)**2 + cxe

def LogisticRegression_w_b(score, labels, l):
    x0 = numpy.zeros(1 + 1)
    x, f, d = scipy.optimize.fmin_l_bfgs_b(logreg_obj_w_b, x0, args=(score, labels, l), approx_grad = True,
                                            maxfun = 15000, factr=1.0)
    w = x[0:-1]
    b = x[-1]

    return w, b

class LinearRegression(object):
    def __init__(self, D, L, lSet,  pca: Optional[List[int]] = None, flag: Optional[bool] = True):
        self. D = D
        self. L = L
        self.type = "Linear Regression"
        self.raw = loadRawFolds()
        self.normalized = loadNormFolds()
        if pca is None:
            self.pca = [D.shape[0]]
        else:
            assert max(pca) <= D.shape[0], f"pca must be smaller than {D.shape[0]}"
            self.pca = pca
        self.print_flag = flag
        self.print_file = "data/Results/LinRegression.txt"
        self.lSet = lSet

    def lr(self, DTR, LTR, DTE, l, prior_t):
        x0 = numpy.zeros(DTR.shape[0] + 1)
        x, f, d = scipy.optimize.fmin_l_bfgs_b(logreg_obj, x0, args=(DTR, LTR, l, prior_t), approx_grad = True)
        w = x[0:DTR.shape[0]]
        b = x[-1]
        scores = numpy.dot(w.T, DTE) + b

        return scores

    def kFold(self, prior_t, folds, l, pca):
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
            LLRsRet = self.lr(DTR, LTR, DTE, l, prior_t)
            LLRs.append(LLRsRet)
        
        LLRs = numpy.hstack(LLRs)

        return LLRs
    
    def plot(self, flag: Optional[bool] = True):
        print("Plotting LR results...")
        f = open(self.print_file, 'r')

        normalized=[]
        raw=[]

        for line in f:
            elements = line.split("|")
            elements =[elem.strip() for elem in elements] 
            if(float(elements[0]) == 0.5):
                if(elements[4]=="Normalized"):
                    normalized.append(( elements[1], float(elements[8].split("=")[1]) ))
                else:
                    raw.append(( elements[1], float(elements[8].split("=")[1]) ))

        nor = numpy.array(normalized,dtype="float")
        raw = numpy.array(raw,dtype="float")

        normalized05=[]
        normalized01 = []
        raw05=[]
        raw01 = []

        for n in nor:
            if(float(n[0]) == 0.5):
                normalized05.append(n[1])
            if (float(n[0]) == 0.1):
                normalized01.append(n[1])
        
        for n in raw:
            if(float(n[0]) == 0.5):
                raw05.append(n[1])
            if (float(n[0]) == 0.1):
                raw01.append(n[1])

        raw05 = numpy.array(raw05)
        raw01 = numpy.array(raw01)

        norm_plot_file = "data/Plots/LinRegression_Norm.png"
        raw_plot_file = "data/Plots/LinRegression_Raw.png"

        plt.plotTwoDCFs(self.lSet, normalized05, normalized01, "λ", "Normalized", norm_plot_file, flag=flag)
        plt.plotTwoDCFs(self.lSet, raw05, raw01, "λ", "Raw", raw_plot_file, flag=flag)
            
    def train(self, prior_t):
        prior_tilde_set = [0.1, 0.5]

        f = open(self.print_file, 'w')

        hyperparameter_list = [(l, i) for l in self.lSet for i in self.pca]

        for l, i in tqdm(hyperparameter_list, desc="Training LR...", ncols=100):
            LLRs = self.kFold(prior_t, self.raw, l, i)

            for prior_tilde in prior_tilde_set:
                ActDCF, minDCF = me.printDCFs(self.D, self.L, LLRs, prior_tilde)
                if self.print_flag:
                    print(f"{prior_t} | {prior_tilde} | {self.type} | Lambda = {l:.2e} | Raw | Uncalibrated | PCA = {i}" + \
                            f"| ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}")
                print(f"{prior_t} | {prior_tilde} | {self.type} | Lambda = {l:.2e} | Raw | Uncalibrated | PCA = {i}" + \
                        f"| ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}", file=f)

            LLRs = self.kFold(prior_t, self.normalized, l, i)
            for prior_tilde in prior_tilde_set:    
                ActDCF, minDCF = me.printDCFs(self.D, self.L, LLRs, prior_tilde)
                if self.print_flag:
                    print(f"{prior_t} | {prior_tilde} | {self.type} | Lambda = {l:.2e} | Normalized | Uncalibrated | PCA = {i}" + \
                        f"| ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}")
                print(f"{prior_t} | {prior_tilde} | {self.type} | Lambda = {l:.2e} | Normalized | Uncalibrated | PCA = {i}" + \
                        f"| ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}", file=f)

    def evaluate(self, prior_t):
        prior_tilde_set = [0.1, 0.5]

        string = str(prior_t).replace(".","")

        file_path = f"data/FinalEvaluation/LinRegression_{string}.txt"

        f = open(file_path, "w")

        D_test, L_test = load_test()
        norm_D_test, norm_L_test = load_norm_test()
        D, L = load_train()
        norm_D, L = load_norm_train()

        hyperparameter_list = [(l, i) for l in self.lSet for i in self.pca]

        #print("result[0] = prior_t | result[1] = prior_tilde | result[2] = model_name | result[3] = pre-processing | result[4] = PCA | result[5] = ActDCF | result[6] = MinDCF")

        for l, i in tqdm(hyperparameter_list, desc="Evaluating LR...", ncols=100):
            D_pca, D_test_pca = apply_PCA(D, D_test, i)
            LLRs = self.lr(D_pca, L, D_test_pca, l, prior_t)

            for prior_tilde in prior_tilde_set:
                ActDCF, minDCF = me.printDCFs(D_test_pca, L_test, LLRs, prior_tilde)
                if self.print_flag:
                    print(f"{prior_t} | {prior_tilde} | {self.type} | Lambda = {l:.2e} | Raw | Uncalibrated | PCA = {i}" + \
                            f"| ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}")
                print(f"{prior_t} | {prior_tilde} | {self.type} | Lambda = {l:.2e} | Raw | Uncalibrated | PCA = {i}" + \
                        f"| ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}", file=f)  

            norm_D_pca, norm_D_test_pca = apply_PCA(norm_D, norm_D_test, i)
            LLRs = self.lr(norm_D_pca, L, norm_D_test_pca, l, prior_t)

            for prior_tilde in prior_tilde_set:    
                ActDCF, minDCF = me.printDCFs(norm_D_test_pca, norm_L_test, LLRs, prior_tilde)
                if self.print_flag:
                    print(f"{prior_t} | {prior_tilde} | {self.type} | Lambda = {l:.2e} | Normalized | Uncalibrated | PCA = {i}" + \
                        f"| ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}")
                print(f"{prior_t} | {prior_tilde} | {self.type} | Lambda = {l:.2e} | Normalized | Uncalibrated | PCA = {i}" + \
                        f"| ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}", file=f)

class QuadraticRegression(object):
    def __init__(self, D, L, lSet,  pca: Optional[List[int]] = None, flag: Optional[bool] = True):
        self. D = D
        self. L = L
        self.type = "Quadratic Regression"
        self.raw = loadRawFolds()
        self.normalized = loadNormFolds()
        if pca is None:
            self.pca = [D.shape[0]]
        else:
            assert max(pca) <= D.shape[0], f"pca must be smaller than {D.shape[0]}"
            self.pca = pca
        self.print_flag = flag
        self.print_file = "data/Results/QuadRegression.txt"
        self.lSet = lSet

    def expandFeature(self, dataset):
        def vecxxT(x):
            x = x[:, None]
            xxT = x.dot(x.T).reshape(x.size**2)
            return xxT
        expanded = numpy.apply_along_axis(vecxxT, 0, dataset)
        return numpy.vstack([expanded, dataset])

    def qr(self, DTR, LTR, DTE, l, prior_t):
        x0 = numpy.zeros(DTR.shape[0] + 1)
        x, f, d = scipy.optimize.fmin_l_bfgs_b(logreg_obj, x0, args=(DTR, LTR, l, prior_t), approx_grad = True)
        w = x[0:DTR.shape[0]]
        b = x[-1]
        scores = numpy.dot(w.T, DTE) + b

        return scores
    
    def kFold(self, prior_t, folds, l, pca):
        LLRs = []
        Predictions = []

        for f in folds:
            DTR = f[0]
            LTR = f[1]
            DTE = f[2]
            LTE = f[3]
            DTR = self.expandFeature(DTR)
            DTE = self.expandFeature(DTE)
            P = dr.PCA_P(DTR, pca)
            DTR = numpy.dot(P.T, DTR)
            DTE = numpy.dot(P.T, DTE)
            LLRsRet = self.qr(DTR, LTR, DTE, l, prior_t)
            LLRs.append(LLRsRet)
        
        LLRs = numpy.hstack(LLRs)

        return LLRs
    
    def plot(self, flag: Optional[bool] = True):
        print("Plotting QR results...")
        f = open(self.print_file, 'r')

        normalized=[]
        raw=[]

        for line in f:
            elements = line.split("|")
            elements =[elem.strip() for elem in elements] 
            if(float(elements[0]) == 0.5):
                if(elements[4]=="Normalized"):
                    normalized.append(( elements[1], float(elements[7].split("=")[1]) ))
                else:
                    raw.append(( elements[1], float(elements[7].split("=")[1]) ))

        nor = numpy.array(normalized,dtype="float")
        raw = numpy.array(raw,dtype="float")

        normalized05=[]
        normalized01 = []
        raw05=[]
        raw01 = []

        for n in nor:
            if(float(n[0]) == 0.5):
                normalized05.append(n[1])
            if (float(n[0]) == 0.1):
                normalized01.append(n[1])
        
        for n in raw:
            if(float(n[0]) == 0.5):
                raw05.append(n[1])
            if (float(n[0]) == 0.1):
                raw01.append(n[1])

        norm_plot_file = "data/Plots/QuadraticRegression_Norm.png"
        raw_plot_file = "data/Plots/QuadraticRegression_Raw.png"

        plt.plotTwoDCFs(self.lSet, normalized05, normalized01, "λ", "Normalized", norm_plot_file, flag=flag)
        plt.plotTwoDCFs(self.lSet, raw05, raw01, "λ", "Raw", raw_plot_file, flag=flag)
        
    def train(self, prior_t):
        prior_tilde_set = [0.1, 0.5]

        f = open(self.print_file, 'w')

        hyperparameter_list = [(l, i) for l in self.lSet for i in self.pca]

        for l, i in tqdm(hyperparameter_list, desc="Training QR...", ncols=100):
            LLRs = self.kFold(prior_t, self.raw, l, i)
            for prior_tilde in prior_tilde_set:
                ActDCF, minDCF = me.printDCFs(self.D, self.L, LLRs, prior_tilde)
                if self.print_flag:    
                    print(f"{prior_t} | {prior_tilde} | {self.type} | Lambda = {l:.2e} | Raw | PCA = {i}" + \
                          f" | ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}")
                print(f"{prior_t} | {prior_tilde} | {self.type} | Lambda = {l:.2e} | Raw | PCA = {i}" + \
                      f" | ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}", file=f)

            LLRs = self.kFold(prior_t, self.normalized, l, i)
            for prior_tilde in prior_tilde_set:    
                ActDCF, minDCF = me.printDCFs(self.D, self.L, LLRs, prior_tilde)
                if self.print_flag:
                    print(f"{prior_t} | {prior_tilde} | {self.type} | Lambda = {l:.2e} | Normalized | PCA = {i}" + \
                          f" | ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}")
                print(f"{prior_t} | {prior_tilde} | {self.type} | Lambda = {l:.2e} | Normalized | PCA = {i}" + \
                      f" | ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}", file=f)


    def evaluate(self, prior_t):
        prior_tilde_set = [0.1, 0.5]

        string = str(prior_t).replace(".","")

        file_path = f"data/FinalEvaluation/QuadRegression_{string}.txt"

        f = open(file_path, "w")

        D_test, L_test = load_test()
        norm_D_test, norm_L_test = load_norm_test()
        D, L = load_train()
        norm_D, L = load_norm_train()

        hyperparameter_list = [(l, i) for l in self.lSet for i in self.pca]

        # As in the k_fold approach, we first expand the features and then apply PCA, this implies that the same
        # procedure has been adopted for both training and evaluation
        
        D = self.expandFeature(D)
        D_test = self.expandFeature(D_test)
        norm_D = self.expandFeature(norm_D)
        norm_D_test = self.expandFeature(norm_D_test)

        for l, i in tqdm(hyperparameter_list, desc="Evaluating QR...", ncols=100):
            D_pca, D_test_pca = apply_PCA(D, D_test, i)
            LLRs = self.qr(D_pca, L, D_test_pca, l, prior_t)

            for prior_tilde in prior_tilde_set:
                ActDCF, minDCF = me.printDCFs(D_test_pca, L_test, LLRs, prior_tilde)
                if self.print_flag:    
                    print(f"{prior_t} | {prior_tilde} | {self.type} | Lambda = {l:.2e} | Raw | PCA = {i}" + \
                          f" | ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}")
                print(f"{prior_t} | {prior_tilde} | {self.type} | Lambda = {l:.2e} | Raw | PCA = {i}" + \
                      f" | ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}", file=f)

            norm_D_pca, norm_D_test_pca = apply_PCA(norm_D, norm_D_test, i)
            LLRs = self.qr(norm_D_pca, L, norm_D_test_pca, l, prior_t)

            for prior_tilde in prior_tilde_set:    
                ActDCF, minDCF = me.printDCFs(norm_D_test_pca, norm_L_test, LLRs, prior_tilde)
                if self.print_flag:
                    print(f"{prior_t} | {prior_tilde} | {self.type} | Lambda = {l:.2e} | Normalized | PCA = {i}" + \
                          f" | ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}")
                print(f"{prior_t} | {prior_tilde} | {self.type} | Lambda = {l:.2e} | Normalized | PCA = {i}" + \
                      f" | ActDCF = {round(ActDCF, 3)} | MinDCF = {round(minDCF,3)}", file=f)

