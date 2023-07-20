import matplotlib.pyplot as plt
import numpy
import scipy.stats as st
from typing import Optional
import math

def plotHist(D, L, string):
    #Ogni riga della matrice è una feature
    D0 = D[:, L == 0]
    D1 = D[:, L == 1]

    hFea = {
        0: 'Feature 1',
        1: 'Feature 2',
        2: 'Feature 3',
        3: 'Feature 4',
        4: 'Feature 5',
        5: 'Feature 6',
    } 

    for i in range(D.shape[0]):
        plt.figure()
        plt.xlabel(hFea[i])
        plt.hist(D0[i, :], bins=100, density=True, alpha=0.4, label='Another Language', color='red')
        plt.hist(D1[i, :], bins=100, density=True, alpha=0.4, label='Considered Language', color='blue')
        plt.legend()
        plt.tight_layout()  # Use with non-default font size to keep axis label inside the figure
        plt.savefig(string + 'hist_%d.png' % i)
        plt.close()

def HeatMapPearson(D, string):
    corr = []
    for i in range(D.shape[0]):
        row = []
        for j in range(D.shape[0]):
            v = st.pearsonr(D[i],D[j])
            row.append(v[0])
        corr.append(row)

    corr = numpy.vstack(corr)

    plt.imshow(corr, cmap="Blues")
    plt.xlabel("Pearson Heat Map")
    plt.savefig(string + "HeatMap.png")
    plt.close()

def plotDCF(x, y):
    #x = numpy.linspace(min(x), max(x), 7)
    plt.plot(x, y, label = "DCF")
    plt.xscale("log")

    #plt.savefig("Plot_LR.pdf")
    plt.show()

def plotTwoDCFs(x, y1, y2, variabile, type, filename: Optional[str] = None, flag: Optional[bool] = True):
    plt.figure()
    plt.plot(x, y1, label = "0.5", color = "r")
    plt.plot(x, y2, label = "0.1", color = "y")
    plt.xlim([min(x), max(x)])
    plt.xscale("log", base = 10)
    plt.legend(["minDCF("r'$\tilde{\pi}$'" = 0.5)", "minDCF("r'$\tilde{\pi}$'" = 0.1)", "minDCF("r'$\tilde{\pi}$'" = 0.9)"])
    
    plt.xlabel(variabile)
    plt.ylabel("MinDCF " + type)

    if filename is not None:
        plt.savefig(filename)

    if flag:
        plt.show()

def plotThreeDCFsRBF(x, y1, y2, y3, variabile, type, filename: Optional[str] = None, flag: Optional[bool] = True):
    plt.figure()
    plt.plot(x, y1, label = "0.5", color = "r")
    plt.plot(x, y2, label = "0.1", color = "y")
    plt.plot(x, y3, label = "0.9", color = "m")
    plt.xlim([min(x), max(x)])
    plt.xscale("log", base = 10)
    plt.legend(["logγ = -3", "logγ = -2", "logγ = -1"])
    
    plt.xlabel(variabile)
    plt.ylabel("MinDCF " + type)

    if filename is not None:
        plt.savefig(filename)

    if flag:
        plt.show()

def plotHistGMM(x, y1, y2, type, filename: Optional[str] = None, flag: Optional[bool] = True):
    f, ax = plt.subplots()

    width = 0.35

    x = numpy.array([int(math.log2(x)) for x in x])

    ax.bar(x - width/2, y1, width)
    ax.bar(x + width/2, y2, width)
    labels = 2**x
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(["Raw", "Normalized"])

    ax.set_xlabel("GMM Components")
    ax.set_ylabel("Min DCF " + type)

    if filename is not None:
        plt.savefig(filename)

    if flag:
        plt.show()