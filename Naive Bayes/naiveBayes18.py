# Justin Choi (jc8mc)
# April 22nd 2018 
# CS 4501: Machine Learning Homework 5 (NBC Text Classification)

import sys
import os
import numpy as np
import nltk 
nltk.download('stopwords')
from sklearn.naive_bayes import MultinomialNB
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

def transfer(fileDj, vocabulary):
    BOWDj = np.zeros((1, len(vocabulary)))
    counter = dict()
    for word in vocabulary:
        counter[word] = 0
    with open(fileDj) as fd:
        for line in fd:
            line = line.replace("loving", "love").replace("loves", "love").replace("loved", "love") 
            tokens = line.split()
            for token in tokens:
                if token in counter:
                    counter[token] += 1
                else:
                    counter["UNK"] += 1

    for i in range(0, len(vocabulary)):
        BOWDj[0][i] = counter[vocabulary[i]]
    return BOWDj

def load_data(Path):
    voc = given_words 
    x_train, y_train = load_data_helper(os.path.join(Path, "training_set"), voc)
    x_test, y_test = load_data_helper(os.path.join(Path, "test_set"), voc)
    return x_train, x_test, y_train, y_test, voc

# helper function
def load_data_helper(path, voc):
    x = np.empty((0, len(voc)), int)
    y = np.empty((0, 1), int)
    label_dict = dict()
    label_dict["pos"] = 1
    label_dict["neg"] = -1
    for dirName, subdirList, fileList in os.walk(path):
        directory = os.path.basename(dirName)
        if directory in label_dict:
            label = label_dict[directory]
            for document in fileList:
                bow = transfer(os.path.join(dirName, document), voc)
                x = np.append(x, bow, axis=0)
                y = np.append(y, label)
    return x, y

def naiveBayesMulFeature_train(x_train, y_train):
    counter = Counter(y_train)
    neg_size = counter[-1]
    x_train_negative = x_train[0:neg_size]
    x_train_positive = x_train[neg_size: len(y_train)]

    theta_pos = (np.sum(x_train_positive, axis=0) + 1).astype("f8") / (np.sum(x_train_positive) + len(x_train[0]))
    theta_neg = (np.sum(x_train_negative, axis=0) + 1).astype("f8") / (np.sum(x_train_negative) + len(x_train[0]))
    return theta_pos, theta_neg


def naiveBayesMulFeature_test(x_test, y_test, theta_pos, theta_neg):
    log_theta_pos = np.log(theta_pos)
    log_theta_neg = np.log(theta_neg)
    pos_predict = np.dot(x_test, log_theta_pos)
    neg_predict = np.dot(x_test, log_theta_neg)
    y_predict = pos_predict - neg_predict
    y_predict[y_predict > 0] = 1
    y_predict[y_predict < 0] = -1
    correct = 0
    total = len(y_test)
    for i in range(0, total):
        if y_predict[i] == y_test[i]:
            correct += 1
    accuracy = float(correct) / total
    return y_predict, accuracy

def naiveBayesMulFeature_sk_MNBC(x_train, y_train, x_test, y_test):
    clf = MultinomialNB()
    clf.fit(x_train, y_train)
    clf.predict(x_test)
    accuracy = clf.score(x_test, y_test)
    return accuracy

def naiveBayesMulFeature_testDirectOne(path, theta_pos, theta_neg, voc):
    index_map = dict()
    index = 0
    for word in voc:
        index_map[word] = index
        index += 1
    log_theta_pos = np.log(theta_pos)
    log_theta_neg = np.log(theta_neg)
    log_prob_pos = 0
    log_prob_neg = 0
    with open(path) as fd:
        for line in fd:
            line = line.replace("loving", "love").replace("loves", "love").replace("loved", "love")
            tokens = line.split()
            for token in tokens:
                if token in index_map:
                    log_prob_pos += log_theta_pos[index_map[token]]
                    log_prob_neg += log_theta_neg[index_map[token]]
                else:
                    log_prob_pos += log_theta_pos[index_map["UNK"]]
                    log_prob_neg += log_theta_neg[index_map["UNK"]]

    if log_prob_pos > log_prob_neg:
        return 1
    else:
        return -1

def naiveBayesMulFeature_testDirect(path, theta_pos, theta_neg, voc):
    y_predict = []
    correct = 0
    total = 0
    label_dict = dict()
    label_dict["pos"] = 1
    label_dict["neg"] = -1
    for dirName, subdirList, fileList in os.walk(path):
        directory = os.path.basename(dirName)
        if directory in label_dict:
            label = label_dict[directory]
            for document in fileList:
                total += 1
                y_predict_one = naiveBayesMulFeature_testDirectOne(os.path.join(dirName, document), theta_pos,
                                                                   theta_neg, voc)
                y_predict.append(y_predict_one)
                if y_predict_one == label:
                    correct += 1

    accuracy = float(correct) / total
    return np.array(y_predict), accuracy

def naiveBayesBernFeature_train(x_train, y_train):
    counter = Counter(y_train)
    neg_size = counter[-1]
    x_train_negative = x_train[0:neg_size]
    x_train_positive = x_train[neg_size: len(y_train)]

    theta_pos_true = np.zeros(len(x_train[0]))
    for i in range(0, len(theta_pos_true)):
        theta_pos_true[i] = np.count_nonzero(x_train_positive[:, i])
    theta_pos_true = (theta_pos_true + 1).astype("f8") / (len(x_train_positive) + 2)

    theta_neg_true = np.zeros(len(x_train[0]))
    for i in range(0, len(theta_neg_true)):
        theta_neg_true[i] = np.count_nonzero(x_train_negative[:, i])
    theta_neg_true = (theta_neg_true + 1).astype("f8") / (len(x_train_negative) + 2)

    return theta_pos_true, theta_neg_true

def naiveBayesBernFeature_test(x_test, y_test, theta_pos_true, theta_neg_true):
    y_predict = []
    for i in range(0, len(x_test)):
        document = x_test[i]
        log_pro_pos_true = 0
        log_pro_neg_true = 0
        for j in range(0, len(document)):
            if document[j] > 0:
                log_pro_pos_true += np.log(theta_pos_true[j])
                log_pro_neg_true += np.log(theta_neg_true[j])
            else:
                log_pro_pos_true += np.log(1 - theta_pos_true[j])
                log_pro_neg_true += np.log(1 - theta_neg_true[j])
        if log_pro_pos_true > log_pro_neg_true:
            y_predict.append(1)
        else:
            y_predict.append(-1)

    correct = 0
    total = len(y_test)
    for i in range(0, total):
        if y_predict[i] == y_test[i]:
            correct += 1
    accuracy = float(correct) / total
    return np.array(y_predict), accuracy

if __name__ == "__main__":
    given_words = ['love', 'wonderful', 'best', 'great', 'superb', 'still', 'beautiful', 'bad', 'worst', 'stupid',
                     'waste', 'boring', '?', '!', 'UNK']

    stop_words = stopwords.words('english')
    more_stop_words = [',', '.', '"', ')', '(', '--', '-', ';', ':', '*']
    stop_words.extend(more_stop_words)
    stemmer = SnowballStemmer("english")
    
    if len(sys.argv) != 3:
        print "Usage: python naiveBayes.py dataSetPath testSetPath"
        sys.exit()

    print "--------------------"
    textDataSetsDirectoryFullPath = sys.argv[1]
    testFileDirectoryFullPath = sys.argv[2]

    Xtrain, Xtest, ytrain, ytest, vocabulary = load_data(textDataSetsDirectoryFullPath)

    thetaPos, thetaNeg = naiveBayesMulFeature_train(Xtrain, ytrain)
    print "thetaPos =", thetaPos
    print "thetaNeg =", thetaNeg
    print "--------------------"

    yPredict, Accuracy = naiveBayesMulFeature_test(Xtest, ytest, thetaPos, thetaNeg)
    print "MNBC classification accuracy =", Accuracy

    Accuracy_sk = naiveBayesMulFeature_sk_MNBC(Xtrain, ytrain, Xtest, ytest)
    print "Sklearn MultinomialNB accuracy =", Accuracy_sk

    yPredict, Accuracy = naiveBayesMulFeature_testDirect(testFileDirectoryFullPath, thetaPos, thetaNeg, vocabulary)
    print "Directly MNBC tesing accuracy =", Accuracy
    print "--------------------"

    thetaPosTrue, thetaNegTrue = naiveBayesBernFeature_train(Xtrain, ytrain)
    print "thetaPosTrue =", thetaPosTrue
    print "thetaNegTrue =", thetaNegTrue
    print "--------------------"

    yPredict, Accuracy = naiveBayesBernFeature_test(Xtest, ytest, thetaPosTrue, thetaNegTrue)
    print "BNBC classification accuracy =", Accuracy
    print "--------------------"
