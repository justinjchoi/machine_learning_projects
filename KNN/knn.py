# Machine Learning HW2-KNN

__author__ = 'Justin Choi (jc8mc)'
import numpy as np
import pandas as pd  
import math
import random
from sklearn.neighbors import KNeighborsClassifier

# file is just a filename, this method read in file contents
# Att: there are many ways to read in one reference dataset, 
# e.g., this template reads in the whole file and put it into one numpy array. 
# (But in HW1, our template actually read the file into two numpy array, one for Xval, the other for Yval. 
# Both ways are correct.) 
def read_csv(file):
    # use pandas to read the txt / csv file seperated by tabs 
    data = pd.read_csv(file, sep='\t')
    # shuffle the data after reading 
    data = data.sample(frac=1).reset_index(drop=True)
    return data

#data is the full training numpy array
#k is the current iteration of cross validation
#kfold is the total number of cross validation folds
def fold(data, k, kfold):
    # now split the features dataset into k-folds 
    split_data = np.array_split(data, kfold)
    # make each training and testing from the split_data for each fold 
    testing = split_data[k]
    training = [x for i,x in enumerate(split_data) if i != k]
    training = np.vstack(training)
    return training, testing
    


#training is the numpy array of training data 
#(you run through each testing point and classify based on the training points)
#testing is a numpy array, use this method to predict 1 or 0 for each of the testing points
#k is the number of neighboring points to take into account when predicting the label
def classify(training, testing, k):
    classified_list = []
    for each_test_row in range(len(testing)): 
        euclidean_distance_list = []
        for each_train_row in range(len(training)): 
            # calculate euclidean distance 
            b1 = testing[:,:len(testing[each_test_row])-1][each_test_row]
            b2 = training[:,:len(training[each_train_row])-1][each_train_row]
            euclidean_distance = math.sqrt((b1-b2).sum() ** 2)
            # get the y value from train 
            y_hat = training[:,len(training[each_train_row])-1][each_train_row]
            euclidean_distance_list.append([euclidean_distance, y_hat])
        # sort the array 
        euclidean_distance_list.sort()
        # slice the array based on k 
        nearest_neighbor = euclidean_distance_list[:k]
        zeros = sum(row.count(0) for row in nearest_neighbor)
        ones = sum(row.count(1) for row in nearest_neighbor)
        if ones > zeros:
            classified_list.append(1)
        else: 
            classified_list.append(0)
    return np.asarray(classified_list, dtype = float)

#predictions is a numpy array of 1s and 0s for the class prediction
#labels is a numpy array of 1s and 0s for the true class label
def calc_accuracy(predictions, labels):
    correct = 0 
    for each_predict in range(len(predictions)): 
        if (predictions[each_predict] == labels[each_predict]):
            correct += 1
    accuracy = float(correct) / len(predictions)
    return accuracy
        


def main():
    filename = "/Users/choi/ML/HW2/Movie_Review_Data.txt"
    kfold = 3
    k = str(input("Provide an odd k value: "))
    while (not k.isdigit()):
        k =str( input("Provide an odd k value: "))
    k = int(k)
    sum = 0
    data = np.asarray(read_csv(filename), dtype=float)
    for i in range(0, kfold):
        training, testing = fold(data, i, kfold)
        predictions = classify(training, testing, k)
        labels = testing[:,-1]
        sum += calc_accuracy(predictions, labels)
    accuracy = sum / kfold
    print(accuracy)


if __name__ == "__main__":
    main()
