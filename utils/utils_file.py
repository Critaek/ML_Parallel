import numpy
import os
from k_fold_utilities.Normalized import NormDTE

norm_test_path = "data/norm_test.npy"
norm_train_path = "data/norm_train.npy"

def mcol(v):
    return v.reshape(v.size, 1)

def vrow(vect):
    return vect.reshape(1, vect.size)

def load_train():
    train = open("data/Train.txt")
    DList = []
    LabelsList = []
    for line in train:
        numbers = line.split(",")[0:-1]
        numbers = mcol(numpy.array([float(i) for i in numbers]))
        DList.append(numbers)
        LabelsList.append(line.split(",")[-1])
    
    D = numpy.hstack(DList) 
    L = numpy.array(LabelsList, dtype=numpy.int32)

    return D, L

def load_test():
    test = open("data/Test.txt")
    DList = []
    LabelsList = []
    for line in test:
        numbers = line.split(",")[0:-1]
        numbers = mcol(numpy.array([float(i) for i in numbers]))
        DList.append(numbers)
        LabelsList.append(line.split(",")[-1])
    
    D = numpy.hstack(DList) 
    L = numpy.array(LabelsList, dtype=numpy.int32)

    return D, L

def load_norm_test():
    D_test, L_test = load_test()

    if not os.path.exists(norm_test_path):    
        D, L = load_train()
        norm_D_test = NormDTE(D, D_test)
        numpy.save(norm_test_path, norm_D_test)

    return numpy.load(norm_test_path, allow_pickle=True), L_test

def load_norm_train():
    D, L = load_train()

    if not os.path.exists(norm_train_path):
        norm_D_train = NormDTE(D, D)
        numpy.save(norm_train_path, norm_D_train)

    return numpy.load(norm_train_path, allow_pickle=True), L
