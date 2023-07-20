import utils.Plot as plt
import numpy

# Just a simple script to extract THE best from some file, just not to do it by hand
# and to plot some results in the evaluation phase

file = "data/FinalEvaluation/SVMPoly_01.txt"

file = open(file, "r")

elements = []

for line in file:
    elem = line.split("|")
    elem = [x.strip() for x in elem]
    elements.append(elem)

el1 = []

for i, el in enumerate(elements):
    if el[5] == "d = 2.0" and el[9] == "PCA = 6" and el[3] == "K = 10.0" and el[4] == "C = 0.01" and el[7] == "Raw" and el[6] == "c = 1.0":
        mindcf = el[-1].split("=")[-1]
        el1.append((i, mindcf, el))

minimo = min(el1, key=lambda x: x[1])
print(minimo)

"""
file = "data/FinalEvaluation/SVMLinear_01.txt"

file = open(file, "r")

elements = []

for line in file:
    elem = line.split("|")
    elem = [x.strip() for x in elem]
    elements.append(elem)

el2 = []

for i, el in enumerate(elements):
    if el[3] == "K = 1.0" and el[7] == "PCA = 6" and el[4] == "C = 0.01":
        mindcf = el[-1].split("=")[-1]
        el2.append((i, mindcf, el))

minimo = min(el2, key=lambda x: x[1])
print(minimo)

file = "data/FinalEvaluation/SVMPoly_01.txt"

file = open(file, "r")

elements = []

for line in file:
    elem = line.split("|")
    elem = [x.strip() for x in elem]
    elements.append(elem)

el3 = []

for i, el in enumerate(elements):
    if el[5] == "d = 2.0" and el[9] == "PCA = 6":
        mindcf = el[-1].split("=")[-1]
        el3.append((i, mindcf, el))

minimo = min(el3, key=lambda x: x[1])
print(minimo)
"""

C_Set = numpy.logspace(-2,0, num = 5)

print("Plotting SVM Poly results...")
f = open("data/FinalEvaluation/SVMPoly_05.txt", "r")
i_MinDCF = []
lines = []

#0.1 | 0.1 | SVM Poly | K = 0.0 | C = 0.01 | d = 2.0 | c = 0.0 | Raw | Uncalibrated | PCA = 10 | ActDCF =1.737 | MinDCF =0.997

for i, line in enumerate(f):
    elements = line.split("|")
    elements =[elem.strip() for elem in elements]
    lines.append(elements)
    MinDCF = elements[11][8:]
    i_MinDCF.append((i, float(elements[0]), MinDCF)) #(indice, prior, mindcf)

i_MinDCF05 = filter(lambda x: x[1] == 0.5, i_MinDCF)
MinDCF = min(i_MinDCF05, key = lambda x: x[2])
#print(MinDCF)
index = MinDCF[0]
#print(lines[index])

Best_K = "K = 10.0"
Best_d = "d = 2.0"
Best_c = "c = 1.0"
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
    PCA = line[9]
    K = line[3]
    d = line[5]
    c = line[6]
    minDCF = float(line[11][8:])

    if (prior_t == 0.5 and Cal == "Uncalibrated"):
        if (K == Best_K and d == Best_d and c == Best_c and PCA == "PCA = 6"):
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

norm_plot_file = f"data/Plots/SVMPoly_{Best_K}_{Best_d}_{Best_c}_Norm_05_FE.png"
raw_plot_file = f"data/Plots/SVMPoly_{Best_K}_{Best_d}_{Best_c}_Raw_05_FE.png"
                    
plt.plotTwoDCFs(C_Set, raw05, raw01, "C", "Raw", raw_plot_file)
plt.plotTwoDCFs(C_Set, normalized05, normalized01, "C", "Normalized", norm_plot_file)