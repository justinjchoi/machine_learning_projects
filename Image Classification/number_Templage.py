#!/usr/bin/env python

import sys
import csv
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.decomposition import RandomizedPCA
from sklearn.linear_model import Perceptron
import numpy as np

train_label = []
train_data = []

test_real_label = []
test_data = []

def loadData(file):
    print "load data"
    data = np.loadtxt(file)
    xData = data[:, 1:256]
    yData = data[:, 0]
    return xData, yData

def knn(train, test):
    fid = open(train)
    tid = open(test)
    for line in fid:
        line = line.strip()
        m = line.split(' ')
        train_label.append(int(float(m[0])))
        train_data.append(m[1:])

    for line in tid:
        line = line.strip()
        m = line.split(' ')
        test_real_label.append(int(float(m[0])))
        test_data.append(m[1:])
    count = 0

    neigh = KNeighborsClassifier(n_neighbors = 1)
    neigh.fit(train_data, train_label)
    y1 = neigh.predict(test_data)

    for i in range(2007):
        if int(float(y1[i])) == test_real_label[i]:
            count += 1
    acc1 = count * 1.0 / 2007

    count = 0

    neigh = KNeighborsClassifier(n_neighbors=5)
    neigh.fit(train_data, train_label)
    y2 = neigh.predict(test_data)

    for i in range(2007):
        if int(float(y2[i])) == test_real_label[i]:
            count += 1
    acc2 = count * 1.0 / 2007

    count = 0

    neigh = KNeighborsClassifier(n_neighbors=10)
    neigh.fit(train_data, train_label)
    y3 = neigh.predict(test_data)

    for i in range(2007):
        if int(float(y3[i])) == test_real_label[i]:
            count += 1
    acc3 = count * 1.0 / 2007

    count = 0
    neigh = KNeighborsClassifier(n_neighbors=20)
    neigh.fit(train_data, train_label)
    y4 = neigh.predict(test_data)

    for i in range(2007):
        if int(float(y4[i])) == test_real_label[i]:
            count += 1
    acc4 = count * 1.0 / 2007

    count = 0
    neigh = KNeighborsClassifier(n_neighbors=200)
    neigh.fit(train_data, train_label)
    y5 = neigh.predict(test_data)

    for i in range(2007):
        if int(float(y5[i])) == test_real_label[i]:
            count += 1
    acc5 = count * 1.0 / 2007

    y = y2
    return y

def neural_net(train, test):
    y = []
    xTrain, yTrain = loadData(train)
    xTest, yTest = loadData(test)
    nN = Perceptron()
    nN.fit(xTrain, yTrain)
    y = nN.predict(xTest)
    testError = 1 - nN.score(xTest, yTest)
    print 'Test error: ' , testError
    return y

def pca_knn(train, test):
    y = []
    xTrain, yTrain = loadData(train)
    xTest, yTest = loadData(test)
    for i in [32, 64, 128] :
        print "n_components", i
        pca = RandomizedPCA(n_components = i, random_state = 1)
        pca.fit(xTrain)
        reducedXTrain = pca.transform(xTrain)
        reducedXTest = pca.transform(xTest)
        kNN = KNeighborsClassifier(n_neighbors = 4, weights = 'distance')
        kNN.fit(reducedXTrain, yTrain)
        y = kNN.predict(reducedXTest)
        testError = 1 - kNN.score(reducedXTest, yTest)
        print 'Test error: ' , testError
        print "sum of explained_variance_ratio_", pca.explained_variance_ratio_.sum()
    return y

if __name__ == '__main__':
    model = sys.argv[1]
    train = sys.argv[2]
    test = sys.argv[3]

    if model == "knn":
        print(knn(train, test))
    elif model == "net":
        print(neural_net(train, test))
    elif model == "pcaknn":
        print(pca_knn(train, test))
    else:
        print("Invalid method selected!")
