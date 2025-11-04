#!/usr/bin/env python
# coding: utf-8

# # Python file on KNN-STROUD
# 

# In[1]:


from Algorithms import KNN
from Algorithms import STROUD
from Algorithms import LOF
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from Evaluations import ConfidenceEstimation
from numpy import genfromtxt
from Graphs import ROC
from matplotlib import pyplot as plt
from scipy.spatial import distance
import random
import tensorflow as tf
import numpy as np
import timeit
from scipy.fft import fft, ifft, fftfreq
from scipy.spatial import distance
from sklearn.neighbors import NearestNeighbors
import scipy.io as sio
import csv
import os
from Algorithms import SVD
from Algorithms import SVDDenoise

# Changes size of plots
plt.rcParams['figure.figsize'] = [15, 5]


# In[2]:


params = {'legend.fontsize': 50,
          'figure.figsize': (45, 15),
         'axes.labelsize': 60,
         'axes.titlesize':70,
         'xtick.labelsize':50,
         'ytick.labelsize':50,
          'lines.linewidth':10,
         'lines.markersize':10}
plt.rcParams.update(params)


# ## Method to get f1 and accuracy score

# In[3]:


# DESCRIPTION:  Calculates the accumulative results of f1 and accuracy due to that fact that ConfidenceEstimation class
# does not already do this.
#
# INPUTS:   (a) list_of_metrics - a list containing several metrics dictionaries
# RETURNS:  (a) results - a dictionary with the accumulative results of f1 and acc in AVG,MIN and MAX forms
# May place into ConfidenceEstimation Later.
def getf1andacc(list_of_metrics):
    f1_list = [item["f1"] for item in list_of_metrics]
    acc_list = [item["acc"] for item in list_of_metrics]
    
    metrics = {
        "f1": f1_list,
        "acc": acc_list
    }
    
    return metrics


# ### Methods to save information

# In[4]:


# DESCRIPTION: Saves the basic information on given Experiment info into separate csv files
def saveToFileSplit(tns, fps, fns, tps, fprs, tprs, location, preName):  
    #try:
    csv_columns = ['tns']
    csv_file = preName + "_tns.csv"   
    with open(location + csv_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(map(lambda x: [x], tns))

    csv_columns = ['fps']
    csv_file = preName + "_fps.csv"   
    with open(location + csv_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(map(lambda x: [x], fps))


    csv_columns = ['fns']
    csv_file = preName + "_fns.csv"   
    with open(location + csv_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(map(lambda x: [x], fns))


    csv_columns = ['tps']
    csv_file = preName + "_tps.csv"   
    with open(location + csv_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(map(lambda x: [x], tps))

    csv_columns = ['fprs']
    csv_file = preName + "_fprs.csv"   
    with open(location + csv_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(map(lambda x: [x], fprs))

    csv_columns = ['tprs']
    csv_file = preName + "_tprs.csv"   
    with open(location + csv_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(map(lambda x: [x], tprs))

    #except IOError:
    #    print("I/O error")


# In[5]:


#Save f1s, accs, auc of each fold.
def saveScores(f1s, accs, auc, precs, recalls, location, preName):
    csv_columns = ['f1s']
    csv_file = preName + "_f1s.csv"   
    with open(location + csv_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(map(lambda x: [x], f1s))
    
    csv_columns = ['accs']
    csv_file = preName + "_accs.csv"   
    with open(location + csv_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(map(lambda x: [x], accs))
    
    csv_columns = ['auc']
    csv_file = preName + "_auc.csv"   
    with open(location + csv_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(map(lambda x: [x], auc))
        
    csv_columns = ['precs']
    csv_file = preName + "_precs.csv"   
    with open(location + csv_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(map(lambda x: [x], precs))
        
    csv_columns = ['recalls']
    csv_file = preName + "_recalls.csv"   
    with open(location + csv_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(map(lambda x: [x], recalls))


# In[6]:


def printandwrite(loc, text, firstLine = False):       
    print(text)
    if firstLine:
        sourceFile = open(loc, 'w')
    else:
        sourceFile = open(loc, 'a')
    print(text, file = sourceFile)
    sourceFile.close()


# In[7]:


def findBestNeighbor(accBests, neighbors):

    accBest = 0
    bestIndex = -1
    
    for index in range(neighbors.shape[0]):
        if accBests[index] > accBest:
            bestIndex = index
            accBest = accBests[index]
            
    return neighbors[bestIndex]


# ## Method to perform KNN with stroud on normal and anomalous datasets

# In[8]:

# Metric calculations for KNN
# calculates all metrics at each threshold point for auc
def getEvaluationKNN(KNN_Scores, ground_truth, numberThresholds=100):
    results = []

    for threshold in np.arange(0, 1 + 1 / numberThresholds, 1 / numberThresholds):
        predictions = []
        for index in range(KNN_Scores.shape[0]):
            if KNN_Scores[index].mean() > threshold:
                predictions.append(1)
            else:
                predictions.append(0)

        predictions = np.array(predictions)

        metrics = ConfidenceEstimation.calculateBasicMetrics(ground_truth, predictions)

        results.append(metrics)

    return results

# Metric calculations for KNN
# calculates all metrics at each threshold point for auc
def getEvaluationKNN_CheckAll(KNN_Scores, ground_truth, numberThresholds=100, maxi=5):
    results = []

    for threshold in np.arange(0, maxi + 1 / numberThresholds, 1 / numberThresholds):
        predictions = []
        for index in range(KNN_Scores.shape[0]):
            if KNN_Scores[index].mean() > threshold:
                predictions.append(1)
            else:
                predictions.append(0)

        predictions = np.array(predictions)

        metrics = ConfidenceEstimation.calculateBasicMetrics(ground_truth, predictions)

        results.append(metrics)

    return results

def getMaxs(lists, maxi):
    locs = []
    for i in range(len(lists)):
        if lists[i] == maxi:
            locs.append(i)
    return locs

def getListGivenLocs(lists, locs):
    new_List = []
    for loc in locs:
        new_List.append(lists[loc])
    return new_List

def KNN(normalTrain, normalTest, anomalyTest, neighbors, normalTrainInst, normalTestInst, anomalyInst, name, pathWrite,
               pathMatrix, pathAUC, pathDist, randomseed = 25, numFold = 1, Info = True, skip=0.001, checkAll=False):
    
    random.seed(randomseed)
    np.random.seed(randomseed)
    tf.random.set_seed(randomseed)
    
    start = timeit.default_timer()
    
    aucs = []
    f1s = []
    accs = []
    precs = []
    recalls = []
    
    fprsies = []
    tprsies = []
    tprs_Mean = []
    
    count = 0
    
    # initialise and specific the number of splits
    #kf = KFold(n_splits=numFold)
    # use the normalTrain to select indexes for training and testing subsets
    #kf.get_n_splits(normalTrain)
    # for each split
    #for train_index, test_index in kf.split(normalTrain):
        
    length = int(normalTest.shape[0])

    # X is the data y are the labels
    # X_train is comprised only of normal observations (obtained from normalTrain)
    # X_test is comprised of two parts X_test_A (only normal observations) and X_test_B (only abnormal observations)
    X_train = normalTrain
    X_test_A = normalTest
    # re-use the same indexes for X_test_B also (this is not a mistake)
    # in this way we make sure that unique observations are used in the test set for each split

    X_test_B = anomalyTest

    X_train_size = X_train.shape[0]
    X_test_A_size = X_test_A.shape[0]
    X_test_B_size = X_test_B.shape[0]
    # time to concatenate test sets
    X_test = np.concatenate((X_test_A, X_test_B))
    # y_train is never used it is just there for reasons of readability
    y_train = [0] * X_train_size
    # concatenate the labels
    y_test = np.concatenate(([0] * X_test_A_size, [1] * X_test_B_size))

    #transformed_training_set = X_train
    #transformed_test_set = X_test
    # compute the strangess scores: the base and test KNN score
    # method
    #base_strangeness_list = KNN.executeKNN(X_train, X_train, neighbors)
    #test_strangeness_list = KNN.executeKNN(X_train, X_test, neighbors)

    nbrs = NearestNeighbors(n_neighbors=neighbors)
    nbrs.fit(X_train)

    distTrain, indexTrain = nbrs.kneighbors(X_train)
    distTest, indexTest = nbrs.kneighbors(X_test)

    mini = np.amin(distTrain)
    maxi = np.amax(distTrain) # Range of Baseline
    #maxi = np.amax(distTest) # Range of Any

    outfile = pathDist + 'distTrain_N=' + str(neighbors) + '_Fold=' + str(count) + '.npy'
    np.save(outfile, distTrain)

    outfile = pathDist + 'distTest_N=' + str(neighbors) + '_Fold=' + str(count) + '.npy'
    np.save(outfile, distTest)

    X_Test_KNNScores = distTest / maxi

    # run threshold test and obtain information
    list_of_metrics = getEvaluationKNN(X_Test_KNNScores, y_test, 1000)

    methodName = "KNN"

    #print("base_strangeness_list")
    #for n in range(len(base_strangeness_list)):
    #    print(str(n) + " " + str(base_strangeness_list[n]))

    #print("test_strangeness_list")
    #for n in range(len(test_strangeness_list)):
    #    print(str(n) + " " + str(test_strangeness_list[n]))

    # compute the p_values
    # Uses the STROUD method
    #p_values = STROUD.executeSTROUD(base_strangeness_list, test_strangeness_list, "false")

    #print("p_values")
    #for n in range(p_values.shape[0]):
    #    print(str(n) + " " + str(p_values[n] * 1000000))

    # compute the metrics (fpr, tpr, auc) for each confidence level   
    #list_of_metrics = ConfidenceEstimation.getEvaluation(p_values, y_test, 0, 1, skip)
    # accumulate metrics
    accumulative_metrics = ConfidenceEstimation.calculateMultipleRoundMetrics(list_of_metrics)

    auc_Score = accumulative_metrics["auc"]
    aucs.append(auc_Score)

    fprs = [item["fall_out"] for item in list_of_metrics]
    tprs = [item["sensitivity"] for item in list_of_metrics]

    fprs = np.array(fprs)
    tprs = np.array(tprs)

    base_fpr = np.linspace(0, 1, fprs.shape[0])
    fprsies.append(fprs)
    tprsies.append(tprs)
    tpr = np.interp(base_fpr, np.flip(fprs), np.flip(tprs))
    tpr[0] = 0.0
    tprs_Mean.append(tpr)

    #f1 Acc Prec and Recall      
    f1sSec = [item["f1"] for item in list_of_metrics]
    accsSec = [item["acc"] for item in list_of_metrics]
    precsSec = [item["precision"] for item in list_of_metrics]
    recallsSec = [item["sensitivity"] for item in list_of_metrics]

    locs = getMaxs(accsSec, max(accsSec))
    
    f1sSec = np.array(f1sSec)
    accsSec = np.array(accsSec)
    precsSec = np.array(getListGivenLocs(precsSec, locs))
    recallsSec = np.array(getListGivenLocs(recallsSec, locs))

    f1s.append(max(f1sSec))
    accs.append(max(accsSec))
    precs.append(max(precsSec))
    recalls.append(max(recallsSec))

    # truth matrix
    tnsSec = [item["tn"] for item in list_of_metrics]
    fpsSec = [item["fp"] for item in list_of_metrics]
    fnsSec = [item["fn"] for item in list_of_metrics]
    tpsSec = [item["tp"] for item in list_of_metrics]

    tnsSec = np.array(tnsSec)
    fpsSec = np.array(fpsSec)
    fnsSec = np.array(fnsSec)
    tpsSec = np.array(tpsSec)

    #print(fprs.shape)
    #print(tprs.shape)
    #print(tnsSec.shape)
    #print(fpsSec.shape)
    #print(fnsSec.shape)
    #print(tpsSec.shape)

    #Save the tns, fps, fns, tps matrix. 
    saveToFileSplit(tnsSec, fpsSec, fnsSec, tpsSec, fprs, tprs, pathMatrix, "n=" + str(neighbors) + "_fold" + str(count))

    #Save f1s, accs, auc of each fold.
    saveScores(f1sSec, accsSec, np.array([auc_Score]), precsSec, recallsSec, pathMatrix, "n=" + str(neighbors) + "_fold" + str(count))      

    # Use this to see AUC graphs
    rocTitle = ""
    aucScoreSave = int(auc_Score *1000)
    if Info:
        ROC.plotROC(fprs, tprs, "red", 10, "ROC", "FP Rate", "TP Rate", rocTitle, 4, auc_Score)
        ROC.plotROC(fprs, tprs, "red", 10, "ROC", "FP Rate", "TP Rate", rocTitle, 4, auc_Score, pathAUC + "n=" + str(neighbors) + "_fold" + str(count) + "AUC=" + str(aucScoreSave))

    count = count+1
    
    if Info: 
        printandwrite(pathWrite, "Method: KNN")
        printandwrite(pathWrite, "Normal Dataset Used: " + normalTrainInst)
        printandwrite(pathWrite, "Size of Training Set: " + str(X_train_size))
        printandwrite(pathWrite, "Test Set Used: " + normalTestInst + " normal along with " + anomalyInst + " anomaly instances")
        printandwrite(pathWrite, "Size of Test Set: " + str(X_test_A_size) + " (normal) " + str(X_test_B_size) + " (anomalous) ")
        printandwrite(pathWrite, "Neighbors: " + str(neighbors))
        printandwrite(pathWrite, "Avg f1 score: " + str(np.mean(f1s)))
        printandwrite(pathWrite, "Max f1 score: " + str(max(f1s)))
        printandwrite(pathWrite, "Min f1 score: " + str(min(f1s)))
        printandwrite(pathWrite, "Avg acc score: " + str(np.mean(accs)))
        printandwrite(pathWrite, "Max acc score: " + str(max(accs)))
        printandwrite(pathWrite, "Min acc score: " + str(min(accs)))   
        printandwrite(pathWrite, "Avg AUC score: " + str(np.mean(aucs)))
        printandwrite(pathWrite, "Max AUC score: " + str(max(aucs)))
        printandwrite(pathWrite, "Min AUC score: " + str(min(aucs)))
        printandwrite(pathWrite, "Avg Precision score: " + str(np.mean(precs)))
        printandwrite(pathWrite, "Max Precision score: " + str(max(precs)))
        printandwrite(pathWrite, "Min Precision score: " + str(min(precs)))
        printandwrite(pathWrite, "Avg Recall score: " + str(np.mean(recalls)))
        printandwrite(pathWrite, "Max Recall score: " + str(max(recalls)))
        printandwrite(pathWrite, "Min Recall score: " + str(min(recalls))) 

    stop = timeit.default_timer()
    
    if Info: 
        printandwrite(pathWrite, 'Time: ' + str(stop - start))
    
    return np.mean(f1s), np.mean(accs), np.mean(aucs), np.mean(precs), np.mean(recalls)

def KNN_AUC(normalTrain, normalTest, anomalyTest, neighbors, normalTrainInst, normalTestInst, anomalyInst, name, pathWrite,
               pathMatrix, pathAUC, pathDist, randomseed = 25, numFold = 1, Info = True, skip=0.001):
    
    random.seed(randomseed)
    np.random.seed(randomseed)
    tf.random.set_seed(randomseed)
    
    start = timeit.default_timer()
    
    aucs = []
    f1s = []
    accs = []
    
    fprsies = []
    tprsies = []
    tprs_Mean = []
    
    count = 0
    
    # initialise and specific the number of splits
    #kf = KFold(n_splits=numFold)
    # use the normalTrain to select indexes for training and testing subsets
    #kf.get_n_splits(normalTrain)
    # for each split
    #for train_index, test_index in kf.split(normalTrain):
        
    length = int(normalTest.shape[0])

    # X is the data y are the labels
    # X_train is comprised only of normal observations (obtained from normalTrain)
    # X_test is comprised of two parts X_test_A (only normal observations) and X_test_B (only abnormal observations)
    X_train = normalTrain
    X_test_A = normalTest
    # re-use the same indexes for X_test_B also (this is not a mistake)
    # in this way we make sure that unique observations are used in the test set for each split

    X_test_B = anomalyTest

    X_train_size = X_train.shape[0]
    X_test_A_size = X_test_A.shape[0]
    X_test_B_size = X_test_B.shape[0]
    # time to concatenate test sets
    X_test = np.concatenate((X_test_A, X_test_B))
    # y_train is never used it is just there for reasons of readability
    y_train = [0] * X_train_size
    # concatenate the labels
    y_test = np.concatenate(([0] * X_test_A_size, [1] * X_test_B_size))

    #transformed_training_set = X_train
    #transformed_test_set = X_test
    # compute the strangess scores: the base and test KNN score
    # method
    #base_strangeness_list = KNN.executeKNN(X_train, X_train, neighbors)
    #test_strangeness_list = KNN.executeKNN(X_train, X_test, neighbors)

    nbrs = NearestNeighbors(n_neighbors=neighbors)
    nbrs.fit(X_train)

    distTrain, indexTrain = nbrs.kneighbors(X_train)
    distTest, indexTest = nbrs.kneighbors(X_test)

    mini = np.amin(distTrain)
    maxi = np.amax(distTrain) # Range of Baseline
    #maxi = np.amax(distTest) # Range of Any

    outfile = pathDist + 'distTrain_N=' + str(neighbors) + '_Fold=' + str(count) + '.npy'
    np.save(outfile, distTrain)

    outfile = pathDist + 'distTest_N=' + str(neighbors) + '_Fold=' + str(count) + '.npy'
    np.save(outfile, distTest)

    X_Test_KNNScores = distTest / maxi

    # run threshold test and obtain information
    list_of_metrics = getEvaluationKNN(X_Test_KNNScores, y_test, 1000)

    methodName = "KNN"

    #print("base_strangeness_list")
    #for n in range(len(base_strangeness_list)):
    #    print(str(n) + " " + str(base_strangeness_list[n]))

    #print("test_strangeness_list")
    #for n in range(len(test_strangeness_list)):
    #    print(str(n) + " " + str(test_strangeness_list[n]))

    # compute the p_values
    # Uses the STROUD method
    #p_values = STROUD.executeSTROUD(base_strangeness_list, test_strangeness_list, "false")

    #print("p_values")
    #for n in range(p_values.shape[0]):
    #    print(str(n) + " " + str(p_values[n] * 1000000))

    # compute the metrics (fpr, tpr, auc) for each confidence level   
    #list_of_metrics = ConfidenceEstimation.getEvaluation(p_values, y_test, 0, 1, skip)
    # accumulate metrics
    accumulative_metrics = ConfidenceEstimation.calculateMultipleRoundMetrics(list_of_metrics)

    auc_Score = accumulative_metrics["auc"]
    aucs.append(auc_Score)

    fprs = [item["fall_out"] for item in list_of_metrics]
    tprs = [item["sensitivity"] for item in list_of_metrics]

    fprs = np.array(fprs)
    tprs = np.array(tprs)

    base_fpr = np.linspace(0, 1, fprs.shape[0])
    fprsies.append(fprs)
    tprsies.append(tprs)
    tpr = np.interp(base_fpr, np.flip(fprs), np.flip(tprs))
    tpr[0] = 0.0
    tprs_Mean.append(tpr)

    #f1 and Acc      
    f1sSec = [item["f1"] for item in list_of_metrics]
    accsSec = [item["acc"] for item in list_of_metrics]

    f1sSec = np.array(f1sSec)
    accsSec = np.array(accsSec)

    f1s.append(max(f1sSec))
    accs.append(max(accsSec)) 

    # truth matrix
    tnsSec = [item["tn"] for item in list_of_metrics]
    fpsSec = [item["fp"] for item in list_of_metrics]
    fnsSec = [item["fn"] for item in list_of_metrics]
    tpsSec = [item["tp"] for item in list_of_metrics]

    tnsSec = np.array(tnsSec)
    fpsSec = np.array(fpsSec)
    fnsSec = np.array(fnsSec)
    tpsSec = np.array(tpsSec)

    #print(fprs.shape)
    #print(tprs.shape)
    #print(tnsSec.shape)
    #print(fpsSec.shape)
    #print(fnsSec.shape)
    #print(tpsSec.shape)

    #Save the tns, fps, fns, tps matrix. 
    saveToFileSplit(tnsSec, fpsSec, fnsSec, tpsSec, fprs, tprs, pathMatrix, "n=" + str(neighbors) + "_fold" + str(count))

    #Save f1s, accs, auc of each fold.
    saveScores(f1sSec, accsSec, np.array([auc_Score]), pathMatrix, "n=" + str(neighbors) + "_fold" + str(count))      

    # Use this to see AUC graphs
    rocTitle = ""
    aucScoreSave = int(auc_Score *1000)
    if Info:
        ROC.plotROC(fprs, tprs, "red", 10, "ROC", "FP Rate", "TP Rate", rocTitle, 4, auc_Score)
        ROC.plotROC(fprs, tprs, "red", 10, "ROC", "FP Rate", "TP Rate", rocTitle, 4, auc_Score, pathAUC + "n=" + str(neighbors) + "_fold" + str(count) + "AUC=" + str(aucScoreSave))

    count = count+1
    
    if Info: 
        printandwrite(pathWrite, "Method: KNN")
        printandwrite(pathWrite, "Normal Dataset Used: " + normalTrainInst)
        printandwrite(pathWrite, "Size of Training Set: " + str(X_train_size))
        printandwrite(pathWrite, "Test Set Used: " + normalTestInst + " normal along with " + anomalyInst + " anomaly instances")
        printandwrite(pathWrite, "Size of Test Set: " + str(X_test_A_size) + " (normal) " + str(X_test_B_size) + " (anomalous) ")
        printandwrite(pathWrite, "Neighbors: " + str(neighbors))
        printandwrite(pathWrite, "Avg f1 score: " + str(np.mean(f1s)))
        printandwrite(pathWrite, "Max f1 score: " + str(max(f1s)))
        printandwrite(pathWrite, "Min f1 score: " + str(min(f1s)))
        printandwrite(pathWrite, "Avg acc score: " + str(np.mean(accs)))
        printandwrite(pathWrite, "Max acc score: " + str(max(accs)))
        printandwrite(pathWrite, "Min acc score: " + str(min(accs)))   
        printandwrite(pathWrite, "Avg AUC score: " + str(np.mean(aucs)))
        printandwrite(pathWrite, "Max AUC score: " + str(max(aucs)))
        printandwrite(pathWrite, "Min AUC score: " + str(min(aucs)))     

    stop = timeit.default_timer()
    
    if Info: 
        printandwrite(pathWrite, 'Time: ' + str(stop - start))
    
    return tprs, fprs, np.mean(aucs)

def KNN_kFold(normalTrain, normalTest, anomalyTest, neighbors, normalTrainInst, normalTestInst, anomalyInst, name, pathWrite,
               pathMatrix, pathAUC, pathDist, randomseed = 25, numFold = 10, Info = True, skip=0.001, checkAll=False, maximum_Dist=-1):
    
    random.seed(randomseed)
    np.random.seed(randomseed)
    tf.random.set_seed(randomseed)
    
    start = timeit.default_timer()
    
    aucs = []
    f1s = []
    accs = []
    precs = []
    recalls = []
    
    fprsies = []
    tprsies = []
    tprs_Mean = []
    
    count = 0
    
    # initialise and specific the number of splits
    kf = KFold(n_splits=numFold)
    # use the normalTrain to select indexes for training and testing subsets
    kf.get_n_splits(normalTrain)
    # for each split
    for train_index, test_index in kf.split(normalTrain):
        
        length = int(test_index.shape[0])
        
        test_index = test_index[0:length]
        
        # X is the data y are the labels
        # X_train is comprised only of normal observations (obtained from normalTrain)
        # X_test is comprised of two parts X_test_A (only normal observations) and X_test_B (only abnormal observations)
        X_train, X_test_A = normalTrain[train_index], normalTest[test_index]
        # re-use the same indexes for X_test_B also (this is not a mistake)
        # in this way we make sure that unique observations are used in the test set for each split
               
        X_test_B = anomalyTest[test_index]
            
        X_train_size = X_train.shape[0]
        X_test_A_size = X_test_A.shape[0]
        X_test_B_size = X_test_B.shape[0]
        # time to concatenate test sets
        X_test = np.concatenate((X_test_A, X_test_B))
        # y_train is never used it is just there for reasons of readability
        y_train = [0] * X_train_size
        # concatenate the labels
        y_test = np.concatenate(([0] * X_test_A_size, [1] * X_test_B_size))
        
        #transformed_training_set = X_train
        #transformed_test_set = X_test
        # compute the strangess scores: the base and test KNN score
        # method
        #base_strangeness_list = KNN.executeKNN(X_train, X_train, neighbors)
        #test_strangeness_list = KNN.executeKNN(X_train, X_test, neighbors)
        
        nbrs = NearestNeighbors(n_neighbors=neighbors)
        nbrs.fit(X_train)

        distTrain, indexTrain = nbrs.kneighbors(X_train)
        distTest, indexTest = nbrs.kneighbors(X_test)

        mini = np.amin(distTrain)
        maxi = np.amax(distTrain) # Range of Baseline
        #maxi = np.amax(distTest) # Range of Any

        outfile = pathDist + 'distTrain_N=' + str(neighbors) + '_Fold=' + str(count) + '.npy'
        np.save(outfile, distTrain)

        outfile = pathDist + 'distTest_N=' + str(neighbors) + '_Fold=' + str(count) + '.npy'
        np.save(outfile, distTest)

        X_Test_KNNScores = distTest / maxi

        # run threshold test and obtain information
        if checkAll:
            #print(maximum_Dist)
            if maximum_Dist > 0:
                maximum_Dist = maximum_Dist
            else:
                maximum_Dist = np.max(np.array(X_Test_KNNScores))
            list_of_metrics = getEvaluationKNN_CheckAll(X_Test_KNNScores, y_test, 1000, maximum_Dist)
            
        else:
            list_of_metrics = getEvaluationKNN(X_Test_KNNScores, y_test, 1000)

        methodName = "KNN"
        
        #print("base_strangeness_list")
        #for n in range(len(base_strangeness_list)):
        #    print(str(n) + " " + str(base_strangeness_list[n]))
            
        #print("test_strangeness_list")
        #for n in range(len(test_strangeness_list)):
        #    print(str(n) + " " + str(test_strangeness_list[n]))
        
        # compute the p_values
        # Uses the STROUD method
        #p_values = STROUD.executeSTROUD(base_strangeness_list, test_strangeness_list, "false")
        
        #print("p_values")
        #for n in range(p_values.shape[0]):
        #    print(str(n) + " " + str(p_values[n] * 1000000))
        
        # compute the metrics (fpr, tpr, auc) for each confidence level   
        #list_of_metrics = ConfidenceEstimation.getEvaluation(p_values, y_test, 0, 1, skip)
        # accumulate metrics
        accumulative_metrics = ConfidenceEstimation.calculateMultipleRoundMetrics(list_of_metrics)
        
        auc_Score = accumulative_metrics["auc"]
        aucs.append(auc_Score)
        
        fprs = [item["fall_out"] for item in list_of_metrics]
        tprs = [item["sensitivity"] for item in list_of_metrics]
        
        fprs = np.array(fprs)
        tprs = np.array(tprs)
        
        base_fpr = np.linspace(0, 1, fprs.shape[0])
        fprsies.append(fprs)
        tprsies.append(tprs)
        tpr = np.interp(base_fpr, np.flip(fprs), np.flip(tprs))
        tpr[0] = 0.0
        tprs_Mean.append(tpr)
        
        #f1 Acc prec recall      
        f1sSec = [item["f1"] for item in list_of_metrics]
        accsSec = [item["acc"] for item in list_of_metrics]
        precsSec = [item["precision"] for item in list_of_metrics]
        recallsSec = [item["sensitivity"] for item in list_of_metrics]
        
        locs = getMaxs(accsSec, max(accsSec))
    
        f1sSec = np.array(f1sSec)
        accsSec = np.array(accsSec)
        precsSec = np.array(getListGivenLocs(precsSec, locs))
        recallsSec = np.array(getListGivenLocs(recallsSec, locs))
        
        f1s.append(max(f1sSec))
        accs.append(max(accsSec))
        precs.append(max(precsSec))
        recalls.append(max(recallsSec))
        
        # truth matrix
        tnsSec = [item["tn"] for item in list_of_metrics]
        fpsSec = [item["fp"] for item in list_of_metrics]
        fnsSec = [item["fn"] for item in list_of_metrics]
        tpsSec = [item["tp"] for item in list_of_metrics]
        
        tnsSec = np.array(tnsSec)
        fpsSec = np.array(fpsSec)
        fnsSec = np.array(fnsSec)
        tpsSec = np.array(tpsSec)
        
        #print(fprs.shape)
        #print(tprs.shape)
        #print(tnsSec.shape)
        #print(fpsSec.shape)
        #print(fnsSec.shape)
        #print(tpsSec.shape)
        
        #Save the tns, fps, fns, tps matrix. 
        saveToFileSplit(tnsSec, fpsSec, fnsSec, tpsSec, fprs, tprs, pathMatrix, "n=" + str(neighbors) + "_fold" + str(count))

        #Save f1s, accs, auc of each fold.
        saveScores(f1sSec, accsSec, np.array([auc_Score]), precsSec, recallsSec, pathMatrix, "n=" + str(neighbors) + "_fold" + str(count))      
        
        # Use this to see AUC graphs
        rocTitle = anomalyInst
        aucScoreSave = int(auc_Score *1000)
        if Info:
            ROC.plotROC(fprs, tprs, "red", 10, "ROC", "FP Rate", "TP Rate", rocTitle, 4, auc_Score)
            ROC.plotROC(fprs, tprs, "red", 10, "ROC", "FP Rate", "TP Rate", rocTitle, 4, auc_Score, pathAUC + "n=" + str(neighbors) + "_fold" + str(count) + "AUC=" + str(aucScoreSave))
                 
        count = count+1
    
    if Info:
        ROC.plotROCMultiple(fprsies, tprsies, aucs, tprs_Mean, "green", 10, "FP Rate", "TP Rate", rocTitle, 4)
        ROC.plotROCMultiple(fprsies, tprsies, aucs, tprs_Mean, "green", 10, "FP Rate", "TP Rate", rocTitle, 4, pathAUC +"_Full")
    
    if Info: 
        printandwrite(pathWrite, "Method: KNN")
        printandwrite(pathWrite, "Normal Dataset Used: " + normalTrainInst)
        printandwrite(pathWrite, "Size of Training Set: " + str(X_train_size))
        printandwrite(pathWrite, "Test Set Used: " + normalTestInst + " normal along with " + anomalyInst + " anomaly instances")
        printandwrite(pathWrite, "Size of Test Set: " + str(X_test_A_size) + " (normal) " + str(X_test_B_size) + " (anomalous) ")
        printandwrite(pathWrite, "Neighbors: " + str(neighbors))
        printandwrite(pathWrite, "Avg f1 score: " + str(np.mean(f1s)))
        printandwrite(pathWrite, "Max f1 score: " + str(max(f1s)))
        printandwrite(pathWrite, "Min f1 score: " + str(min(f1s)))
        printandwrite(pathWrite, "Avg acc score: " + str(np.mean(accs)))
        printandwrite(pathWrite, "Max acc score: " + str(max(accs)))
        printandwrite(pathWrite, "Min acc score: " + str(min(accs)))   
        printandwrite(pathWrite, "Avg AUC score: " + str(np.mean(aucs)))
        printandwrite(pathWrite, "Max AUC score: " + str(max(aucs)))
        printandwrite(pathWrite, "Min AUC score: " + str(min(aucs)))
        printandwrite(pathWrite, "Avg Precision score: " + str(np.mean(precs)))
        printandwrite(pathWrite, "Max Precision score: " + str(max(precs)))
        printandwrite(pathWrite, "Min Precision score: " + str(min(precs)))
        printandwrite(pathWrite, "Avg Recall score: " + str(np.mean(recalls)))
        printandwrite(pathWrite, "Max Recall score: " + str(max(recalls)))
        printandwrite(pathWrite, "Min Recall score: " + str(min(recalls)))

    stop = timeit.default_timer()
    
    if Info: 
        printandwrite(pathWrite, 'Time: ' + str(stop - start))
    
    return np.mean(f1s), np.mean(accs), np.mean(aucs), np.mean(precs), np.mean(recalls)


# In[ ]:


# Considers multiple datasets to train on.
def KNN_kFold_Train2(normalTrain1, normalTrain2, normalTest, anomalyTest, neighbors, normalTrainInst, normalTestInst, anomalyInst, name, pathWrite,
               pathMatrix, pathAUC, pathDist, randomseed = 25, numFold = 10, Info = True):
    
    random.seed(randomseed)
    np.random.seed(randomseed)
    tf.random.set_seed(randomseed)
    
    start = timeit.default_timer()
    
    aucs = []
    f1s = []
    accs = []
    
    fprsies = []
    tprsies = []
    tprs_Mean = []
    
    count = 0
    
    # initialise and specific the number of splits
    kf = KFold(n_splits=numFold)
    # use the normalTrain to select indexes for training and testing subsets
    kf.get_n_splits(normalTrain1)
    # for each split
    for train_index, test_index in kf.split(normalTrain1):
        
        length = int(test_index.shape[0])
        train_index_Split = train_index[0:int(train_index.shape[0])]
        test_index_Split = test_index[0:int(test_index.shape[0]/2)]
        
        test_index = test_index[0:length]
        
        # X is the data y are the labels
        # X_train is comprised only of normal observations (obtained from normalTrain)
        # X_test is comprised of two parts X_test_A (only normal observations) and X_test_B (only abnormal observations)
        #X_train, X_test_A = normalTrain[train_index], normalTest[test_index]
        X_test_A = np.concatenate((normalTrain1[test_index_Split], normalTest[test_index_Split]))
        
        X_train1 = normalTrain1[train_index_Split]
        X_train2 = normalTrain2[train_index_Split]
        # re-use the same indexes for X_test_B also (this is not a mistake)
        # in this way we make sure that unique observations are used in the test set for each split
               
        X_test_B = anomalyTest[test_index]
            
        X_train_size = X_train1.shape[0]
        X_test_A_size = X_test_A.shape[0]
        X_test_B_size = X_test_B.shape[0]
        # time to concatenate test sets
        X_test = np.concatenate((X_test_A, X_test_B))
        # y_train is never used it is just there for reasons of readability
        #y_train = [0] * X_train_size
        # concatenate the labels
        y_test = np.concatenate(([0] * X_test_A_size, [1] * X_test_B_size))
        
        #transformed_training_set = X_train
        #transformed_test_set = X_test
        # compute the strangess scores: the base and test KNN score
        # method
        base_strangeness_list_1 = KNN.executeKNN(X_train1, X_train1, neighbors)
        base_strangeness_list_2 = KNN.executeKNN(X_train2, X_train2, neighbors)
        
        test_strangeness_list_1 = KNN.executeKNN(X_train1, X_test, neighbors)
        test_strangeness_list_2 = KNN.executeKNN(X_train2, X_test, neighbors)
        
        # compute the p_values
        # Uses the STROUD method
        p_values_test_1 = STROUD.executeSTROUD(base_strangeness_list_1, test_strangeness_list_1, "false")
        p_values_test_2 = STROUD.executeSTROUD(base_strangeness_list_2, test_strangeness_list_2, "false")
        
        p_values_test_1 = np.array(p_values_test_1)
        p_values_test_2 = np.array(p_values_test_2)
        
        p_values_test = np.concatenate((p_values_test_1, p_values_test_2))
        p_values = np.reshape(p_values_test,(2,-1))
        
        # compute the metrics (fpr, tpr, auc) for each confidence level   
        list_of_metrics = ConfidenceEstimation.getEvaluationMultiTrain(p_values, y_test, 0, 1, 0.001)
        # accumulate metrics
        accumulative_metrics = ConfidenceEstimation.calculateMultipleRoundMetrics(list_of_metrics)
        
        auc_Score = accumulative_metrics["auc"]
        aucs.append(auc_Score)
        
        fprs = [item["fall_out"] for item in list_of_metrics]
        tprs = [item["sensitivity"] for item in list_of_metrics]
        
        fprs = np.array(fprs)
        tprs = np.array(tprs)
        
        
        base_fpr = np.linspace(0, 1, 1000)
        fprsies.append(fprs)
        tprsies.append(tprs)
        tpr = np.interp(base_fpr, np.flip(fprs), np.flip(tprs))
        tpr[0] = 0.0
        tprs_Mean.append(tpr)
        
    
        #f1 and Acc      
        f1sSec = [item["f1"] for item in list_of_metrics]
        accsSec = [item["acc"] for item in list_of_metrics]
        
        f1sSec = np.array(f1sSec)
        accsSec = np.array(accsSec)
        
        f1s.append(max(f1sSec))
        accs.append(max(accsSec)) 
        
        # truth matrix
        tnsSec = [item["tn"] for item in list_of_metrics]
        fpsSec = [item["fp"] for item in list_of_metrics]
        fnsSec = [item["fn"] for item in list_of_metrics]
        tpsSec = [item["tp"] for item in list_of_metrics]
        
        tnsSec = np.array(tnsSec)
        fpsSec = np.array(fpsSec)
        fnsSec = np.array(fnsSec)
        tpsSec = np.array(tpsSec)
        
        #Save the tns, fps, fns, tps matrix. 
        saveToFileSplit(tnsSec, fpsSec, fnsSec, tpsSec, fprs, tprs, pathMatrix, "n=" + str(neighbors) + "_fold" + str(count))

        #Save f1s, accs, auc of each fold.
        saveScores(f1sSec, accsSec, np.array([auc_Score]), pathMatrix, "n=" + str(neighbors) + "_fold" + str(count))      
        
        # Use this to see AUC graphs
        rocTitle = anomalyInst
        aucScoreSave = int(auc_Score *1000)
        if Info:
            ROC.plotROC(fprs, tprs, "red", 10, "ROC", "FP Rate", "TP Rate", rocTitle, 4, auc_Score)
            ROC.plotROC(fprs, tprs, "red", 10, "ROC", "FP Rate", "TP Rate", rocTitle, 4, auc_Score, pathAUC + "n=" + str(neighbors) + "_fold" + str(count) + "AUC=" + str(aucScoreSave))
                 
        count = count+1
        
    
    if Info:
        ROC.plotROCMultiple(fprsies, tprsies, aucs, tprs_Mean, "green", 10, "FP Rate", "TP Rate", rocTitle, 4)
        ROC.plotROCMultiple(fprsies, tprsies, aucs, tprs_Mean, "green", 10, "FP Rate", "TP Rate", rocTitle, 4, pathAUC +"_Full")
    
    if Info: 
        printandwrite(pathWrite, "Method: KNN-STROUD")
        printandwrite(pathWrite, "Normal Dataset Used: " + normalTrainInst)
        printandwrite(pathWrite, "Size of Training Set: " + str(X_train_size))
        printandwrite(pathWrite, "Test Set Used: " + normalTestInst + " normal along with " + anomalyInst + " anomaly instances")
        printandwrite(pathWrite, "Size of Test Set: " + str(X_test_A_size) + " (normal) " + str(X_test_B_size) + " (anomalous) ")
        printandwrite(pathWrite, "Neighbors: " + str(neighbors))
        printandwrite(pathWrite, "Avg f1 score: " + str(np.mean(f1s)))
        printandwrite(pathWrite, "Max f1 score: " + str(max(f1s)))
        printandwrite(pathWrite, "Min f1 score: " + str(min(f1s)))
        printandwrite(pathWrite, "Avg acc score: " + str(np.mean(accs)))
        printandwrite(pathWrite, "Max acc score: " + str(max(accs)))
        printandwrite(pathWrite, "Min acc score: " + str(min(accs)))   
        printandwrite(pathWrite, "Avg AUC score: " + str(np.mean(aucs)))
        printandwrite(pathWrite, "Max AUC score: " + str(max(aucs)))
        printandwrite(pathWrite, "Min AUC score: " + str(min(aucs)))     

    stop = timeit.default_timer()
    
    if Info: 
        printandwrite(pathWrite, 'Time: ' + str(stop - start))
    
    return np.mean(f1s), np.mean(accs), np.mean(aucs)


# In[ ]:


# Considers multiple datasets to train on.
def KNN_kFold_norm2_anom1(normalTrain1, normalTrain2, normalTest, anomalyTest, neighbors, normalTrainInst, normalTestInst, anomalyInst, name, pathWrite,
               pathMatrix, pathAUC, pathDist, randomseed = 25, numFold = 10, Info = True):
    
    random.seed(randomseed)
    np.random.seed(randomseed)
    tf.random.set_seed(randomseed)
    
    start = timeit.default_timer()
    
    aucs = []
    f1s = []
    accs = []
    
    fprsies = []
    tprsies = []
    tprs_Mean = []
    
    count = 0
    
    # initialise and specific the number of splits
    kf = KFold(n_splits=numFold)
    # use the normalTrain to select indexes for training and testing subsets
    kf.get_n_splits(normalTrain1)
    # for each split
    for train_index, test_index in kf.split(normalTrain1):
        
        length = int(test_index.shape[0])
        
        test_index = test_index[0:length]
        
        # X is the data y are the labels
        # X_train is comprised only of normal observations (obtained from normalTrain)
        # X_test is comprised of two parts X_test_A (only normal observations) and X_test_B (only abnormal observations)
        #X_train, X_test_A = normalTrain[train_index], normalTest[test_index]
        X_test_A = np.concatenate((normalTrain1[test_index], normalTest[test_index]))
        
        X_train1 = normalTrain1[train_index]
        X_train2 = normalTrain2[train_index]
        # re-use the same indexes for X_test_B also (this is not a mistake)
        # in this way we make sure that unique observations are used in the test set for each split
               
        X_test_B = anomalyTest[test_index]
            
        X_train_size = X_train1.shape[0]
        X_test_A_size = X_test_A.shape[0]
        X_test_B_size = X_test_B.shape[0]
        # time to concatenate test sets
        X_test = np.concatenate((X_test_A, X_test_B))
        # y_train is never used it is just there for reasons of readability
        #y_train = [0] * X_train_size
        # concatenate the labels
        y_test = np.concatenate(([0] * X_test_A_size, [1] * X_test_B_size))
        
        #transformed_training_set = X_train
        #transformed_test_set = X_test
        # compute the strangess scores: the base and test KNN score
        # method
        base_strangeness_list_1 = KNN.executeKNN(X_train1, X_train1, neighbors)
        base_strangeness_list_2 = KNN.executeKNN(X_train2, X_train2, neighbors)
        
        test_strangeness_list_1 = KNN.executeKNN(X_train1, X_test, neighbors)
        test_strangeness_list_2 = KNN.executeKNN(X_train2, X_test, neighbors)
        
        # compute the p_values
        # Uses the STROUD method
        p_values_test_1 = STROUD.executeSTROUD(base_strangeness_list_1, test_strangeness_list_1, "false")
        p_values_test_2 = STROUD.executeSTROUD(base_strangeness_list_2, test_strangeness_list_2, "false")
        
        p_values_test_1 = np.array(p_values_test_1)
        p_values_test_2 = np.array(p_values_test_2)
        
        p_values_test = np.concatenate((p_values_test_1, p_values_test_2))
        p_values = np.reshape(p_values_test,(2,-1))
        
        # compute the metrics (fpr, tpr, auc) for each confidence level   
        list_of_metrics = ConfidenceEstimation.getEvaluationMultiTrain(p_values, y_test, 0, 1, 0.001)
        # accumulate metrics
        accumulative_metrics = ConfidenceEstimation.calculateMultipleRoundMetrics(list_of_metrics)
        
        auc_Score = accumulative_metrics["auc"]
        aucs.append(auc_Score)
        
        fprs = [item["fall_out"] for item in list_of_metrics]
        tprs = [item["sensitivity"] for item in list_of_metrics]
        
        fprs = np.array(fprs)
        tprs = np.array(tprs)
        
        
        base_fpr = np.linspace(0, 1, 1000)
        fprsies.append(fprs)
        tprsies.append(tprs)
        tpr = np.interp(base_fpr, np.flip(fprs), np.flip(tprs))
        tpr[0] = 0.0
        tprs_Mean.append(tpr)
        
    
        #f1 and Acc      
        f1sSec = [item["f1"] for item in list_of_metrics]
        accsSec = [item["acc"] for item in list_of_metrics]
        
        f1sSec = np.array(f1sSec)
        accsSec = np.array(accsSec)
        
        f1s.append(max(f1sSec))
        accs.append(max(accsSec)) 
        
        # truth matrix
        tnsSec = [item["tn"] for item in list_of_metrics]
        fpsSec = [item["fp"] for item in list_of_metrics]
        fnsSec = [item["fn"] for item in list_of_metrics]
        tpsSec = [item["tp"] for item in list_of_metrics]
        
        tnsSec = np.array(tnsSec)
        fpsSec = np.array(fpsSec)
        fnsSec = np.array(fnsSec)
        tpsSec = np.array(tpsSec)
        
        #Save the tns, fps, fns, tps matrix. 
        saveToFileSplit(tnsSec, fpsSec, fnsSec, tpsSec, fprs, tprs, pathMatrix, "n=" + str(neighbors) + "_fold" + str(count))

        #Save f1s, accs, auc of each fold.
        saveScores(f1sSec, accsSec, np.array([auc_Score]), pathMatrix, "n=" + str(neighbors) + "_fold" + str(count))      
        
        # Use this to see AUC graphs
        rocTitle = anomalyInst
        aucScoreSave = int(auc_Score *1000)
        if Info:
            ROC.plotROC(fprs, tprs, "red", 10, "ROC", "FP Rate", "TP Rate", rocTitle, 4, auc_Score)
            ROC.plotROC(fprs, tprs, "red", 10, "ROC", "FP Rate", "TP Rate", rocTitle, 4, auc_Score, pathAUC + "n=" + str(neighbors) + "_fold" + str(count) + "AUC=" + str(aucScoreSave))
                 
        count = count+1
        
    
    if Info:
        ROC.plotROCMultiple(fprsies, tprsies, aucs, tprs_Mean, "green", 10, "FP Rate", "TP Rate", rocTitle, 4)
        ROC.plotROCMultiple(fprsies, tprsies, aucs, tprs_Mean, "green", 10, "FP Rate", "TP Rate", rocTitle, 4, pathAUC +"_Full")
    
    if Info: 
        printandwrite(pathWrite, "Method: KNN-STROUD")
        printandwrite(pathWrite, "Normal Dataset Used: " + normalTrainInst)
        printandwrite(pathWrite, "Size of Training Set: " + str(X_train_size))
        printandwrite(pathWrite, "Test Set Used: " + normalTestInst + " normal along with " + anomalyInst + " anomaly instances")
        printandwrite(pathWrite, "Size of Test Set: " + str(X_test_A_size) + " (normal) " + str(X_test_B_size) + " (anomalous) ")
        printandwrite(pathWrite, "Neighbors: " + str(neighbors))
        printandwrite(pathWrite, "Avg f1 score: " + str(np.mean(f1s)))
        printandwrite(pathWrite, "Max f1 score: " + str(max(f1s)))
        printandwrite(pathWrite, "Min f1 score: " + str(min(f1s)))
        printandwrite(pathWrite, "Avg acc score: " + str(np.mean(accs)))
        printandwrite(pathWrite, "Max acc score: " + str(max(accs)))
        printandwrite(pathWrite, "Min acc score: " + str(min(accs)))   
        printandwrite(pathWrite, "Avg AUC score: " + str(np.mean(aucs)))
        printandwrite(pathWrite, "Max AUC score: " + str(max(aucs)))
        printandwrite(pathWrite, "Min AUC score: " + str(min(aucs)))     

    stop = timeit.default_timer()
    
    if Info: 
        printandwrite(pathWrite, 'Time: ' + str(stop - start))
    
    return np.mean(f1s), np.mean(accs), np.mean(aucs)


# In[ ]:


# Considers multiple datasets to train on.
def KNN_kFold_norm2_anom2(normalTrain1, normalTrain2, normalTest, anomalyTest1, anomalyTest2, neighbors, normalTrainInst, normalTestInst, anomalyInst, name, pathWrite,
               pathMatrix, pathAUC, pathDist, randomseed = 25, numFold = 10, Info = True):
    
    random.seed(randomseed)
    np.random.seed(randomseed)
    tf.random.set_seed(randomseed)
    
    start = timeit.default_timer()
    
    aucs = []
    f1s = []
    accs = []
    
    fprsies = []
    tprsies = []
    tprs_Mean = []
    
    count = 0
    
    # initialise and specific the number of splits
    kf = KFold(n_splits=numFold)
    # use the normalTrain to select indexes for training and testing subsets
    kf.get_n_splits(normalTrain1)
    # for each split
    for train_index, test_index in kf.split(normalTrain1):
        
        length = int(test_index.shape[0])
        
        test_index = test_index[0:length]
        
        # X is the data y are the labels
        # X_train is comprised only of normal observations (obtained from normalTrain)
        # X_test is comprised of two parts X_test_A (only normal observations) and X_test_B (only abnormal observations)
        #X_train, X_test_A = normalTrain[train_index], normalTest[test_index]
        X_test_A = np.concatenate((normalTrain1[test_index], normalTest[test_index]))
        
        X_train1 = normalTrain1[train_index]
        X_train2 = normalTrain2[train_index]
        # re-use the same indexes for X_test_B also (this is not a mistake)
        # in this way we make sure that unique observations are used in the test set for each split
               
        X_test_B = np.concatenate((anomalyTest1[test_index], anomalyTest2[test_index]))
            
        X_train_size = X_train1.shape[0]
        X_test_A_size = X_test_A.shape[0]
        X_test_B_size = X_test_B.shape[0]
        # time to concatenate test sets
        X_test = np.concatenate((X_test_A, X_test_B))
        # y_train is never used it is just there for reasons of readability
        #y_train = [0] * X_train_size
        # concatenate the labels
        y_test = np.concatenate(([0] * X_test_A_size, [1] * X_test_B_size))
        
        #transformed_training_set = X_train
        #transformed_test_set = X_test
        # compute the strangess scores: the base and test KNN score
        # method
        base_strangeness_list_1 = KNN.executeKNN(X_train1, X_train1, neighbors)
        base_strangeness_list_2 = KNN.executeKNN(X_train2, X_train2, neighbors)
        
        test_strangeness_list_1 = KNN.executeKNN(X_train1, X_test, neighbors)
        test_strangeness_list_2 = KNN.executeKNN(X_train2, X_test, neighbors)
        
        # compute the p_values
        # Uses the STROUD method
        p_values_test_1 = STROUD.executeSTROUD(base_strangeness_list_1, test_strangeness_list_1, "false")
        p_values_test_2 = STROUD.executeSTROUD(base_strangeness_list_2, test_strangeness_list_2, "false")
        
        p_values_test_1 = np.array(p_values_test_1)
        p_values_test_2 = np.array(p_values_test_2)
        
        p_values_test = np.concatenate((p_values_test_1, p_values_test_2))
        p_values = np.reshape(p_values_test,(2,-1))
        
        # compute the metrics (fpr, tpr, auc) for each confidence level   
        list_of_metrics = ConfidenceEstimation.getEvaluationMultiTrain(p_values, y_test, 0, 1, 0.001)
        # accumulate metrics
        accumulative_metrics = ConfidenceEstimation.calculateMultipleRoundMetrics(list_of_metrics)
        
        auc_Score = accumulative_metrics["auc"]
        aucs.append(auc_Score)
        
        fprs = [item["fall_out"] for item in list_of_metrics]
        tprs = [item["sensitivity"] for item in list_of_metrics]
        
        fprs = np.array(fprs)
        tprs = np.array(tprs)
        
        
        base_fpr = np.linspace(0, 1, 1000)
        fprsies.append(fprs)
        tprsies.append(tprs)
        tpr = np.interp(base_fpr, np.flip(fprs), np.flip(tprs))
        tpr[0] = 0.0
        tprs_Mean.append(tpr)
        
    
        #f1 and Acc      
        f1sSec = [item["f1"] for item in list_of_metrics]
        accsSec = [item["acc"] for item in list_of_metrics]
        
        f1sSec = np.array(f1sSec)
        accsSec = np.array(accsSec)
        
        f1s.append(max(f1sSec))
        accs.append(max(accsSec)) 
        
        # truth matrix
        tnsSec = [item["tn"] for item in list_of_metrics]
        fpsSec = [item["fp"] for item in list_of_metrics]
        fnsSec = [item["fn"] for item in list_of_metrics]
        tpsSec = [item["tp"] for item in list_of_metrics]
        
        tnsSec = np.array(tnsSec)
        fpsSec = np.array(fpsSec)
        fnsSec = np.array(fnsSec)
        tpsSec = np.array(tpsSec)
        
        #Save the tns, fps, fns, tps matrix. 
        saveToFileSplit(tnsSec, fpsSec, fnsSec, tpsSec, fprs, tprs, pathMatrix, "n=" + str(neighbors) + "_fold" + str(count))

        #Save f1s, accs, auc of each fold.
        saveScores(f1sSec, accsSec, np.array([auc_Score]), pathMatrix, "n=" + str(neighbors) + "_fold" + str(count))      
        
        # Use this to see AUC graphs
        rocTitle = anomalyInst
        aucScoreSave = int(auc_Score *1000)
        if Info:
            ROC.plotROC(fprs, tprs, "red", 10, "ROC", "FP Rate", "TP Rate", rocTitle, 4, auc_Score)
            ROC.plotROC(fprs, tprs, "red", 10, "ROC", "FP Rate", "TP Rate", rocTitle, 4, auc_Score, pathAUC + "n=" + str(neighbors) + "_fold" + str(count) + "AUC=" + str(aucScoreSave))
                 
        count = count+1
        
    
    if Info:
        ROC.plotROCMultiple(fprsies, tprsies, aucs, tprs_Mean, "green", 10, "FP Rate", "TP Rate", rocTitle, 4)
        ROC.plotROCMultiple(fprsies, tprsies, aucs, tprs_Mean, "green", 10, "FP Rate", "TP Rate", rocTitle, 4, pathAUC +"_Full")
    
    if Info: 
        printandwrite(pathWrite, "Method: KNN-STROUD")
        printandwrite(pathWrite, "Normal Dataset Used: " + normalTrainInst)
        printandwrite(pathWrite, "Size of Training Set: " + str(X_train_size))
        printandwrite(pathWrite, "Test Set Used: " + normalTestInst + " normal along with " + anomalyInst + " anomaly instances")
        printandwrite(pathWrite, "Size of Test Set: " + str(X_test_A_size) + " (normal) " + str(X_test_B_size) + " (anomalous) ")
        printandwrite(pathWrite, "Neighbors: " + str(neighbors))
        printandwrite(pathWrite, "Avg f1 score: " + str(np.mean(f1s)))
        printandwrite(pathWrite, "Max f1 score: " + str(max(f1s)))
        printandwrite(pathWrite, "Min f1 score: " + str(min(f1s)))
        printandwrite(pathWrite, "Avg acc score: " + str(np.mean(accs)))
        printandwrite(pathWrite, "Max acc score: " + str(max(accs)))
        printandwrite(pathWrite, "Min acc score: " + str(min(accs)))   
        printandwrite(pathWrite, "Avg AUC score: " + str(np.mean(aucs)))
        printandwrite(pathWrite, "Max AUC score: " + str(max(aucs)))
        printandwrite(pathWrite, "Min AUC score: " + str(min(aucs)))     

    stop = timeit.default_timer()
    
    if Info: 
        printandwrite(pathWrite, 'Time: ' + str(stop - start))
    
    return np.mean(f1s), np.mean(accs), np.mean(aucs)


# In[9]:


def KNN_kFoldSVD(normalTrain, normalTest, anomalyTest, neighbors, normalTrainInst, normalTestInst, anomalyInst, name, pathWrite,
               pathMatrix, pathAUC, pathDist, randomseed = 25, numFold = 10, Info = True):
    
    random.seed(randomseed)
    np.random.seed(randomseed)
    tf.random.set_seed(randomseed)
    
    start = timeit.default_timer()
    
    aucs = []
    f1s = []
    accs = []
    
    count = 0
    
    # initialise and specific the number of splits
    kf = KFold(n_splits=numFold)
    # use the normalTrain to select indexes for training and testing subsets
    kf.get_n_splits(normalTrain)
    # for each split
    for train_index, test_index in kf.split(normalTrain):
        
        length = int(test_index.shape[0]/2)
        
        test_index = test_index[0:length]
        
        # X is the data y are the labels
        # X_train is comprised only of normal observations (obtained from normalTrain)
        # X_test is comprised of two parts X_test_A (only normal observations) and X_test_B (only abnormal observations)
        X_train, X_test_A = normalTrain[train_index], normalTest[test_index]
        # re-use the same indexes for X_test_B also (this is not a mistake)
        # in this way we make sure that unique observations are used in the test set for each split
               
        X_test_B = anomalyTest[test_index]
        
        X_train_size = X_train.shape[0]
        X_test_A_size = X_test_A.shape[0]
        X_test_B_size = X_test_B.shape[0]
        # time to concatenate test sets
        X_test = np.concatenate((X_test_A, X_test_B))
        # y_train is never used it is just there for reasons of readability
        y_train = [0] * X_train_size
        # concatenate the labels
        y_test = np.concatenate(([0] * X_test_A_size, [1] * X_test_B_size))
        
        cutP = SVD.findCutPoint(X_test, False)
        

        if Info:
            print("Training Set")
        transformed_training_set = SVD.SVDdenoise(X_train, Info)    
        if Info:
            print("Test Set")
        transformed_test_set = SVD.SVDdenoise(X_test, Info)
        
        # compute the strangess scores: the base and test KNN score
        # method
        base_strangeness_list = KNN.executeKNN(transformed_training_set, transformed_training_set, neighbors)
        test_strangeness_list = KNN.executeKNN(transformed_training_set, transformed_test_set, neighbors)
        
        # compute the p_values
        # Uses the STROUD method
        p_values = STROUD.executeSTROUD(base_strangeness_list, test_strangeness_list, "false")
        
        # compute the metrics (fpr, tpr, auc) for each confidence level
        list_of_metrics = ConfidenceEstimation.getEvaluation(p_values, y_test, 0, 1, 0.001)
        # accumulate metrics
        accumulative_metrics = ConfidenceEstimation.calculateMultipleRoundMetrics(list_of_metrics)
        
        auc_Score = accumulative_metrics["auc"]
        aucs.append(auc_Score)
        
        fprs = [item["fall_out"] for item in list_of_metrics]
        tprs = [item["sensitivity"] for item in list_of_metrics]
        
        fprs = np.array(fprs)
        tprs = np.array(tprs)
    
        #f1 and Acc      
        f1sSec = [item["f1"] for item in list_of_metrics]
        accsSec = [item["acc"] for item in list_of_metrics]
        
        f1sSec = np.array(f1sSec)
        accsSec = np.array(accsSec)
        
        f1s.append(max(f1sSec))
        accs.append(max(accsSec)) 
        
        # truth matrix
        tnsSec = [item["tn"] for item in list_of_metrics]
        fpsSec = [item["fp"] for item in list_of_metrics]
        fnsSec = [item["fn"] for item in list_of_metrics]
        tpsSec = [item["tp"] for item in list_of_metrics]
        
        tnsSec = np.array(tnsSec)
        fpsSec = np.array(fpsSec)
        fnsSec = np.array(fnsSec)
        tpsSec = np.array(tpsSec)
        
        #Save the tns, fps, fns, tps matrix. 
        saveToFileSplit(tnsSec, fpsSec, fnsSec, tpsSec, fprs, tprs, pathMatrix, "n=" + str(neighbors) + "_fold" + str(count))

        #Save f1s, accs, auc of each fold.
        saveScores(f1sSec, accsSec, np.array([auc_Score]), pathMatrix, "n=" + str(neighbors) + "_fold" + str(count))      
        
        # Use this to see AUC graphs
        rocTitle = anomalyInst
        aucScoreSave = int(auc_Score *1000)
        if Info:
            ROC.plotROC(fprs, tprs, "red", 10, "ROC", "FP Rate", "TP Rate", rocTitle, 4, auc_Score)
            ROC.plotROC(fprs, tprs, "red", 10, "ROC", "FP Rate", "TP Rate", rocTitle, 4, auc_Score, pathAUC + "n=" + str(neighbors) + "_fold" + str(count) + "AUC=" + str(aucScoreSave))
                 
        count = count+1
    
    if Info:
        printandwrite(pathWrite, "Method: KNN-STROUD")
        printandwrite(pathWrite, "Normal Dataset Used: " + normalTrainInst)
        printandwrite(pathWrite, "Size of Training Set: " + str(X_train_size))
        printandwrite(pathWrite, "Test Set Used: " + normalTestInst + " normal along with " + anomalyInst + " anomaly instances")
        printandwrite(pathWrite, "Size of Test Set: " + str(X_test_A_size) + " (normal) " + str(X_test_B_size) + " (anomalous) ")
        printandwrite(pathWrite, "Neighbors: " + str(neighbors))
        printandwrite(pathWrite, "Avg f1 score: " + str(np.mean(f1s)))
        printandwrite(pathWrite, "Max f1 score: " + str(max(f1s)))
        printandwrite(pathWrite, "Min f1 score: " + str(min(f1s)))
        printandwrite(pathWrite, "Avg acc score: " + str(np.mean(accs)))
        printandwrite(pathWrite, "Max acc score: " + str(max(accs)))
        printandwrite(pathWrite, "Min acc score: " + str(min(accs)))   
        printandwrite(pathWrite, "Avg AUC score: " + str(np.mean(aucs)))
        printandwrite(pathWrite, "Max AUC score: " + str(max(aucs)))
        printandwrite(pathWrite, "Min AUC score: " + str(min(aucs)))     

    stop = timeit.default_timer()
    if Info:
        printandwrite(pathWrite, 'Time: ' + str(stop - start))
    
    return np.mean(f1s), np.mean(accs), np.mean(aucs)


# In[ ]:


def KNN_kFoldVHI(normalTrain, normalTest, anomalyTest, neighbors, normalTrainInst, normalTestInst, anomalyInst, name, pathWrite,
               pathMatrix, pathAUC, pathDist, randomseed = 25, numFold = 10, Info = True):
    
    random.seed(randomseed)
    np.random.seed(randomseed)
    tf.random.set_seed(randomseed)
    
    start = timeit.default_timer()
    
    aucs = []
    f1s = []
    accs = []
    
    count = 0
    
    # initialise and specific the number of splits
    kf = KFold(n_splits=numFold)
    # use the normalTrain to select indexes for training and testing subsets
    kf.get_n_splits(normalTrain)
    # for each split
    for train_index, test_index in kf.split(normalTrain):
        
        length = int(test_index.shape[0]/2)
        
        test_index = test_index[0:length]
        
        # X is the data y are the labels
        # X_train is comprised only of normal observations (obtained from normalTrain)
        # X_test is comprised of two parts X_test_A (only normal observations) and X_test_B (only abnormal observations)
        X_train, X_test_A = normalTrain[train_index], normalTest[test_index]
        # re-use the same indexes for X_test_B also (this is not a mistake)
        # in this way we make sure that unique observations are used in the test set for each split
               
        X_test_B = anomalyTest[test_index]
        
        X_train_size = X_train.shape[0]
        X_test_A_size = X_test_A.shape[0]
        X_test_B_size = X_test_B.shape[0]
        # time to concatenate test sets
        X_test = np.concatenate((X_test_A, X_test_B))
        # y_train is never used it is just there for reasons of readability
        y_train = [0] * X_train_size
        # concatenate the labels
        y_test = np.concatenate(([0] * X_test_A_size, [1] * X_test_B_size))
        
        cutPTrain = SVD.findCutPoint(X_train, False)
        cutPTest = SVD.findCutPoint(X_test, False)

        new_v = SVDDenoise.DeduceVhiFilter(X_train, cutPTrain)
        if Info:
            print("Training set VHI, cutting point: " + str(cutPTrain) + ", new_v:")
            print(new_v)
        transformed_training_set = SVDDenoise.ApplyVhiFilter(X_train, new_v)
        
        new_v = SVDDenoise.DeduceVhiFilter(X_test, cutPTest)
        if Info:
            print("Testing set VHI, cutting point: " + str(cutPTest) + ", new_v:")
            print(new_v)
        transformed_test_set = SVDDenoise.ApplyVhiFilter(X_test, new_v)
        
        # compute the strangess scores: the base and test KNN score
        # method
        base_strangeness_list = KNN.executeKNN(transformed_training_set, transformed_training_set, neighbors)
        test_strangeness_list = KNN.executeKNN(transformed_training_set, transformed_test_set, neighbors)
        
        # compute the p_values
        # Uses the STROUD method
        p_values = STROUD.executeSTROUD(base_strangeness_list, test_strangeness_list, "false")
        
        # compute the metrics (fpr, tpr, auc) for each confidence level
        list_of_metrics = ConfidenceEstimation.getEvaluation(p_values, y_test, 0, 1, 0.001)
        # accumulate metrics
        accumulative_metrics = ConfidenceEstimation.calculateMultipleRoundMetrics(list_of_metrics)
        
        auc_Score = accumulative_metrics["auc"]
        aucs.append(auc_Score)
        
        fprs = [item["fall_out"] for item in list_of_metrics]
        tprs = [item["sensitivity"] for item in list_of_metrics]
        
        fprs = np.array(fprs)
        tprs = np.array(tprs)
    
        #f1 and Acc      
        f1sSec = [item["f1"] for item in list_of_metrics]
        accsSec = [item["acc"] for item in list_of_metrics]
        
        f1sSec = np.array(f1sSec)
        accsSec = np.array(accsSec)
        
        f1s.append(max(f1sSec))
        accs.append(max(accsSec)) 
        
        # truth matrix
        tnsSec = [item["tn"] for item in list_of_metrics]
        fpsSec = [item["fp"] for item in list_of_metrics]
        fnsSec = [item["fn"] for item in list_of_metrics]
        tpsSec = [item["tp"] for item in list_of_metrics]
        
        tnsSec = np.array(tnsSec)
        fpsSec = np.array(fpsSec)
        fnsSec = np.array(fnsSec)
        tpsSec = np.array(tpsSec)
        
        #Save the tns, fps, fns, tps matrix. 
        saveToFileSplit(tnsSec, fpsSec, fnsSec, tpsSec, fprs, tprs, pathMatrix, "n=" + str(neighbors) + "_fold" + str(count))

        #Save f1s, accs, auc of each fold.
        saveScores(f1sSec, accsSec, np.array([auc_Score]), pathMatrix, "n=" + str(neighbors) + "_fold" + str(count))      
        
        # Use this to see AUC graphs
        rocTitle = anomalyInst
        aucScoreSave = int(auc_Score *1000)
        if Info:
            ROC.plotROC(fprs, tprs, "red", 10, "ROC", "FP Rate", "TP Rate", rocTitle, 4, auc_Score)
            ROC.plotROC(fprs, tprs, "red", 10, "ROC", "FP Rate", "TP Rate", rocTitle, 4, auc_Score, pathAUC + "n=" + str(neighbors) + "_fold" + str(count) + "AUC=" + str(aucScoreSave))
                 
        count = count+1
    
    if Info:
        printandwrite(pathWrite, "Method: KNN-STROUD")
        printandwrite(pathWrite, "Normal Dataset Used: " + normalTrainInst)
        printandwrite(pathWrite, "Size of Training Set: " + str(X_train_size))
        printandwrite(pathWrite, "Test Set Used: " + normalTestInst + " normal along with " + anomalyInst + " anomaly instances")
        printandwrite(pathWrite, "Size of Test Set: " + str(X_test_A_size) + " (normal) " + str(X_test_B_size) + " (anomalous) ")
        printandwrite(pathWrite, "Neighbors: " + str(neighbors))
        printandwrite(pathWrite, "Avg f1 score: " + str(np.mean(f1s)))
        printandwrite(pathWrite, "Max f1 score: " + str(max(f1s)))
        printandwrite(pathWrite, "Min f1 score: " + str(min(f1s)))
        printandwrite(pathWrite, "Avg acc score: " + str(np.mean(accs)))
        printandwrite(pathWrite, "Max acc score: " + str(max(accs)))
        printandwrite(pathWrite, "Min acc score: " + str(min(accs)))   
        printandwrite(pathWrite, "Avg AUC score: " + str(np.mean(aucs)))
        printandwrite(pathWrite, "Max AUC score: " + str(max(aucs)))
        printandwrite(pathWrite, "Min AUC score: " + str(min(aucs)))     

    stop = timeit.default_timer()
    if Info:
        printandwrite(pathWrite, 'Time: ' + str(stop - start))
    
    return np.mean(f1s), np.mean(accs), np.mean(aucs)


# In[10]:


def KNN_kFoldSVD_Tran(normalTrain, normalTest, anomalyTest, neighbors, normalTrainInst, normalTestInst, anomalyInst, name, pathWrite,
               pathMatrix, pathAUC, pathDist, cleanDataset, cut, randomseed = 25, numFold = 10, Info = True):
    
    random.seed(randomseed)
    np.random.seed(randomseed)
    tf.random.set_seed(randomseed)
    
    start = timeit.default_timer()
    
    aucs = []
    f1s = []
    accs = []
    
    count = 0
    
    # initialise and specific the number of splits
    kf = KFold(n_splits=numFold)
    # use the normalTrain to select indexes for training and testing subsets
    kf.get_n_splits(normalTrain)
    # for each split
    for train_index, test_index in kf.split(normalTrain):
        
        length = int(test_index.shape[0]/2)
        
        test_index = test_index[0:length]
        
        # X is the data y are the labels
        # X_train is comprised only of normal observations (obtained from normalTrain)
        # X_test is comprised of two parts X_test_A (only normal observations) and X_test_B (only abnormal observations)
        X_train, X_test_A = normalTrain[train_index], normalTest[test_index]
        # re-use the same indexes for X_test_B also (this is not a mistake)
        # in this way we make sure that unique observations are used in the test set for each split
               
        X_test_B = anomalyTest[test_index]
        
        X_train_size = X_train.shape[0]
        X_test_A_size = X_test_A.shape[0]
        X_test_B_size = X_test_B.shape[0]
        # time to concatenate test sets
        X_test = np.concatenate((X_test_A, X_test_B))
        # y_train is never used it is just there for reasons of readability
        y_train = [0] * X_train_size
        # concatenate the labels
        y_test = np.concatenate(([0] * X_test_A_size, [1] * X_test_B_size))

        
        new_s = SVDDenoise.DeduceSigmaFilter(cleanDataset, cut)
        if Info:
            print("Training set SVD, cutting point: " + str(cut) + ", new_s:")
            print(new_s)
        transformed_training_set = SVDDenoise.ApplySigmaFilter(X_train, new_s)       
        
        new_s = SVDDenoise.DeduceSigmaFilter(cleanDataset, cut)
        if Info:
            print("Testing set SVD, cutting point: " + str(cut) + ", new_s:")
            print(new_s)
        transformed_test_set = SVDDenoise.ApplySigmaFilter(X_test, new_s)
        

        '''    
        if Info:
            print("Σ saved: ", sigma)
        if Info:
            print("Training set SVD")
        transformed_training_set = SVD.SVDdenoiseCP_LOAD_S(X_train, sigma, cutPTrain, Info)
        if Info:
            print("Test set SVD")
        transformed_test_set = SVD.SVDdenoiseCP_LOAD_S(X_test, sigma, cutPTest, Info)
        '''
        
        # compute the strangess scores: the base and test KNN score
        # method
        base_strangeness_list = KNN.executeKNN(transformed_training_set, transformed_training_set, neighbors)
        test_strangeness_list = KNN.executeKNN(transformed_training_set, transformed_test_set, neighbors)
        
        # compute the p_values
        # Uses the STROUD method
        p_values = STROUD.executeSTROUD(base_strangeness_list, test_strangeness_list, "false")
        
        # compute the metrics (fpr, tpr, auc) for each confidence level
        list_of_metrics = ConfidenceEstimation.getEvaluation(p_values, y_test, 0, 1, 0.001)
        # accumulate metrics
        accumulative_metrics = ConfidenceEstimation.calculateMultipleRoundMetrics(list_of_metrics)
        
        auc_Score = accumulative_metrics["auc"]
        aucs.append(auc_Score)
        
        fprs = [item["fall_out"] for item in list_of_metrics]
        tprs = [item["sensitivity"] for item in list_of_metrics]
        
        fprs = np.array(fprs)
        tprs = np.array(tprs)
    
        #f1 and Acc      
        f1sSec = [item["f1"] for item in list_of_metrics]
        accsSec = [item["acc"] for item in list_of_metrics]
        
        f1sSec = np.array(f1sSec)
        accsSec = np.array(accsSec)
        
        f1s.append(max(f1sSec))
        accs.append(max(accsSec)) 
        
        # truth matrix
        tnsSec = [item["tn"] for item in list_of_metrics]
        fpsSec = [item["fp"] for item in list_of_metrics]
        fnsSec = [item["fn"] for item in list_of_metrics]
        tpsSec = [item["tp"] for item in list_of_metrics]
        
        tnsSec = np.array(tnsSec)
        fpsSec = np.array(fpsSec)
        fnsSec = np.array(fnsSec)
        tpsSec = np.array(tpsSec)
        
        #Save the tns, fps, fns, tps matrix. 
        saveToFileSplit(tnsSec, fpsSec, fnsSec, tpsSec, fprs, tprs, pathMatrix, "n=" + str(neighbors) + "_fold" + str(count))

        #Save f1s, accs, auc of each fold.
        saveScores(f1sSec, accsSec, np.array([auc_Score]), pathMatrix, "n=" + str(neighbors) + "_fold" + str(count))      
        
        # Use this to see AUC graphs
        rocTitle = anomalyInst
        aucScoreSave = int(auc_Score *1000)
        if Info:
            ROC.plotROC(fprs, tprs, "red", 10, "ROC", "FP Rate", "TP Rate", rocTitle, 4, auc_Score)
            ROC.plotROC(fprs, tprs, "red", 10, "ROC", "FP Rate", "TP Rate", rocTitle, 4, auc_Score, pathAUC + "n=" + str(neighbors) + "_fold" + str(count) + "AUC=" + str(aucScoreSave))
                 
        count = count+1
    
    if Info:
        printandwrite(pathWrite, "Method: KNN-STROUD")
        printandwrite(pathWrite, "Normal Dataset Used: " + normalTrainInst)
        printandwrite(pathWrite, "Size of Training Set: " + str(X_train_size))
        printandwrite(pathWrite, "Test Set Used: " + normalTestInst + " normal along with " + anomalyInst + " anomaly instances")
        printandwrite(pathWrite, "Size of Test Set: " + str(X_test_A_size) + " (normal) " + str(X_test_B_size) + " (anomalous) ")
        printandwrite(pathWrite, "Neighbors: " + str(neighbors))
        printandwrite(pathWrite, "Avg f1 score: " + str(np.mean(f1s)))
        printandwrite(pathWrite, "Max f1 score: " + str(max(f1s)))
        printandwrite(pathWrite, "Min f1 score: " + str(min(f1s)))
        printandwrite(pathWrite, "Avg acc score: " + str(np.mean(accs)))
        printandwrite(pathWrite, "Max acc score: " + str(max(accs)))
        printandwrite(pathWrite, "Min acc score: " + str(min(accs)))   
        printandwrite(pathWrite, "Avg AUC score: " + str(np.mean(aucs)))
        printandwrite(pathWrite, "Max AUC score: " + str(max(aucs)))
        printandwrite(pathWrite, "Min AUC score: " + str(min(aucs)))     

    stop = timeit.default_timer()
    if Info:
        printandwrite(pathWrite, 'Time: ' + str(stop - start))
    
    return np.mean(f1s), np.mean(accs), np.mean(aucs)


# In[11]:


def KNN_kFoldVHI_Tran(normalTrain, normalTest, anomalyTest, neighbors, normalTrainInst, normalTestInst, anomalyInst, name, pathWrite,
               pathMatrix, pathAUC, pathDist, cleanDataset, cut, randomseed = 25, numFold = 10, Info = True):
    
    random.seed(randomseed)
    np.random.seed(randomseed)
    tf.random.set_seed(randomseed)
    
    start = timeit.default_timer()
    
    aucs = []
    f1s = []
    accs = []
    
    count = 0
    
    # initialise and specific the number of splits
    kf = KFold(n_splits=numFold)
    # use the normalTrain to select indexes for training and testing subsets
    kf.get_n_splits(normalTrain)
    # for each split
    for train_index, test_index in kf.split(normalTrain):
        
        length = int(test_index.shape[0]/2)
        
        test_index = test_index[0:length]
        
        # X is the data y are the labels
        # X_train is comprised only of normal observations (obtained from normalTrain)
        # X_test is comprised of two parts X_test_A (only normal observations) and X_test_B (only abnormal observations)
        X_train, X_test_A = normalTrain[train_index], normalTest[test_index]
        # re-use the same indexes for X_test_B also (this is not a mistake)
        # in this way we make sure that unique observations are used in the test set for each split
               
        X_test_B = anomalyTest[test_index]
        
        X_train_size = X_train.shape[0]
        X_test_A_size = X_test_A.shape[0]
        X_test_B_size = X_test_B.shape[0]
        # time to concatenate test sets
        X_test = np.concatenate((X_test_A, X_test_B))
        # y_train is never used it is just there for reasons of readability
        y_train = [0] * X_train_size
        # concatenate the labels
        y_test = np.concatenate(([0] * X_test_A_size, [1] * X_test_B_size))
        
        cutPTrain = SVD.findCutPoint(X_train, False)
        cutPTest = SVD.findCutPoint(X_test, False)
        
        new_v = SVDDenoise.DeduceVhiFilter(cleanDataset, cut)
        if Info:
            print("Training set VHI, cutting point: " + str(cut) + ", new_v:")
            print(new_v)
        transformed_training_set = SVDDenoise.ApplyVhiFilter(X_train, new_v)
        
        new_v = SVDDenoise.DeduceVhiFilter(cleanDataset, cut)
        if Info:
            print("Testing set VHI, cutting point: " + str(cut) + ", new_v:")
            print(new_v)
        transformed_test_set = SVDDenoise.ApplyVhiFilter(X_test, new_v)
        
        # compute the strangess scores: the base and test KNN score
        # method
        base_strangeness_list = KNN.executeKNN(transformed_training_set, transformed_training_set, neighbors)
        test_strangeness_list = KNN.executeKNN(transformed_training_set, transformed_test_set, neighbors)
        
        # compute the p_values
        # Uses the STROUD method
        p_values = STROUD.executeSTROUD(base_strangeness_list, test_strangeness_list, "false")
        
        # compute the metrics (fpr, tpr, auc) for each confidence level
        list_of_metrics = ConfidenceEstimation.getEvaluation(p_values, y_test, 0, 1, 0.001)
        # accumulate metrics
        accumulative_metrics = ConfidenceEstimation.calculateMultipleRoundMetrics(list_of_metrics)
        
        auc_Score = accumulative_metrics["auc"]
        aucs.append(auc_Score)
        
        fprs = [item["fall_out"] for item in list_of_metrics]
        tprs = [item["sensitivity"] for item in list_of_metrics]
        
        fprs = np.array(fprs)
        tprs = np.array(tprs)
    
        #f1 and Acc      
        f1sSec = [item["f1"] for item in list_of_metrics]
        accsSec = [item["acc"] for item in list_of_metrics]
        
        f1sSec = np.array(f1sSec)
        accsSec = np.array(accsSec)
        
        f1s.append(max(f1sSec))
        accs.append(max(accsSec)) 
        
        # truth matrix
        tnsSec = [item["tn"] for item in list_of_metrics]
        fpsSec = [item["fp"] for item in list_of_metrics]
        fnsSec = [item["fn"] for item in list_of_metrics]
        tpsSec = [item["tp"] for item in list_of_metrics]
        
        tnsSec = np.array(tnsSec)
        fpsSec = np.array(fpsSec)
        fnsSec = np.array(fnsSec)
        tpsSec = np.array(tpsSec)
        
        #Save the tns, fps, fns, tps matrix. 
        saveToFileSplit(tnsSec, fpsSec, fnsSec, tpsSec, fprs, tprs, pathMatrix, "n=" + str(neighbors) + "_fold" + str(count))

        #Save f1s, accs, auc of each fold.
        saveScores(f1sSec, accsSec, np.array([auc_Score]), pathMatrix, "n=" + str(neighbors) + "_fold" + str(count))      
        
        # Use this to see AUC graphs
        rocTitle = anomalyInst
        aucScoreSave = int(auc_Score *1000)
        if Info:
            ROC.plotROC(fprs, tprs, "red", 10, "ROC", "FP Rate", "TP Rate", rocTitle, 4, auc_Score)
            ROC.plotROC(fprs, tprs, "red", 10, "ROC", "FP Rate", "TP Rate", rocTitle, 4, auc_Score, pathAUC + "n=" + str(neighbors) + "_fold" + str(count) + "AUC=" + str(aucScoreSave))
                 
        count = count+1
    
    if Info:
        printandwrite(pathWrite, "Method: KNN-STROUD")
        printandwrite(pathWrite, "Normal Dataset Used: " + normalTrainInst)
        printandwrite(pathWrite, "Size of Training Set: " + str(X_train_size))
        printandwrite(pathWrite, "Test Set Used: " + normalTestInst + " normal along with " + anomalyInst + " anomaly instances")
        printandwrite(pathWrite, "Size of Test Set: " + str(X_test_A_size) + " (normal) " + str(X_test_B_size) + " (anomalous) ")
        printandwrite(pathWrite, "Neighbors: " + str(neighbors))
        printandwrite(pathWrite, "Avg f1 score: " + str(np.mean(f1s)))
        printandwrite(pathWrite, "Max f1 score: " + str(max(f1s)))
        printandwrite(pathWrite, "Min f1 score: " + str(min(f1s)))
        printandwrite(pathWrite, "Avg acc score: " + str(np.mean(accs)))
        printandwrite(pathWrite, "Max acc score: " + str(max(accs)))
        printandwrite(pathWrite, "Min acc score: " + str(min(accs)))   
        printandwrite(pathWrite, "Avg AUC score: " + str(np.mean(aucs)))
        printandwrite(pathWrite, "Max AUC score: " + str(max(aucs)))
        printandwrite(pathWrite, "Min AUC score: " + str(min(aucs)))     

    stop = timeit.default_timer()
    if Info:
        printandwrite(pathWrite, 'Time: ' + str(stop - start))
    
    return np.mean(f1s), np.mean(accs), np.mean(aucs)


# In[12]:


# Run multiple experiments.
def runExperiment(normal, normalTest, anomalyTest, normalTrainInst, normalTestInst, anomalyInst, name, path,
                  neighbors = np.array([3,5,10,25,50,75,100]), numFold = 10, skip = 0.001, show_Info = True, checkAll=False, maximum_Dist=-1):
    
    normalTrain = normal
    normalTest = normalTest
    anomalyTest = anomalyTest
    
    pathWrite = path + name + ".txt"
    pathMatrix = path + "TruthMatrix/"
    pathAUC = path + "AUCGraphs/"
    pathDist = path + "Distances/"
    
    # Check whether the specified path exists or not
    isExist = os.path.exists(path)
    if not isExist:
      # Create a new directory because it does not exist 
      os.makedirs(path)
    
    # Check whether the specified path exists or not
    isExist = os.path.exists(pathMatrix)
    if not isExist:
      # Create a new directory because it does not exist 
      os.makedirs(pathMatrix)
    
    # Check whether the specified path exists or not
    isExist = os.path.exists(pathAUC)
    if not isExist:
      # Create a new directory because it does not exist 
      os.makedirs(pathAUC)
        
    # Check whether the specified path exists or not
    isExist = os.path.exists(pathDist)
    if not isExist:
      # Create a new directory because it does not exist 
      os.makedirs(pathDist)
    
    f1Bests = []
    accBests = []
    aucBests = []
    precBests = []
    recallBests = []

    if show_Info:
        printandwrite(pathWrite, "Beginning Tests...",True)
                  
    for neighbor in neighbors:
        
        f1Best = 0
        accBest = 0
        aucBest = 0
        
        if numFold > 1:
        #printandwrite(pathWrite, "=======================TEST ON NEIGHBOR = " + str(neighbor) + "=======================")
        
            f1Best, accBest, aucBest, precBest, recallBest = KNN_kFold(normalTrain, normalTest, anomalyTest, neighbor, normalTrainInst, normalTestInst,
                                              anomalyInst, name, pathWrite, pathMatrix, pathAUC, pathDist, numFold = numFold, Info = False, skip=skip, checkAll=checkAll, maximum_Dist=maximum_Dist)
            
        else:
            f1Best, accBest, aucBest, precBest, recallBest = KNN(normalTrain, normalTest, anomalyTest, neighbor, normalTrainInst, normalTestInst,
                                              anomalyInst, name, pathWrite, pathMatrix, pathAUC, pathDist, numFold = numFold, Info = False, skip= skip, checkAll=checkAll, maximum_Dist=maximum_Dist)
        
        f1Bests.append(f1Best)
        accBests.append(accBest)
        aucBests.append(aucBest)
        precBests.append(precBest)
        recallBests.append(recallBest)
        
        #printandwrite(pathWrite, " ")
        
    f1Bests = np.array(f1Bests)
    accBests = np.array(accBests)
    aucBests = np.array(aucBests)
    precBests = np.array(precBests)
    recallBests = np.array(recallBests)
    
    outfile = path + 'f1s.npy'
    np.save(outfile, f1Bests)
    
    outfile = path + 'accs.npy'
    np.save(outfile, accBests)
    
    outfile = path + 'aucs.npy'
    np.save(outfile, aucBests)
    
    outfile = path + 'prec.npy'
    np.save(outfile, precBests)
    
    outfile = path + 'recall.npy'
    np.save(outfile, recallBests)
      
    f1Best = max(f1Bests)
    accBest = max(accBests)
    aucBest = max(aucBests)
    precBest = max(precBests)
    recallBest = max(recallBests)
    
    accBestFind = 0
    bestIndex = -1
    
    BestNeighbor = findBestNeighbor(accBests, neighbors)
    
    if numFold > 1:
        if show_Info:
            f1Best, accBest, aucBest, precBest, recallBest = KNN_kFold(normalTrain, normalTest, anomalyTest, BestNeighbor, normalTrainInst, normalTestInst,
                                                  anomalyInst, name, pathWrite, pathMatrix, pathAUC, pathDist, numFold = numFold, Info = True, skip=skip, checkAll=checkAll, maximum_Dist=maximum_Dist)
        else:
            f1Best, accBest, aucBest, precBest, recallBest = KNN_kFold(normalTrain, normalTest, anomalyTest, BestNeighbor, normalTrainInst, normalTestInst,
                                                  anomalyInst, name, pathWrite, pathMatrix, pathAUC, pathDist, numFold = numFold, Info = False, skip=skip, checkAll=checkAll, maximum_Dist=maximum_Dist)
     
    else:
        if show_Info:
            f1Best, accBest, aucBest, precBest, recallBest = KNN(normalTrain, normalTest, anomalyTest, BestNeighbor, normalTrainInst, normalTestInst,
                                                  anomalyInst, name, pathWrite, pathMatrix, pathAUC, pathDist, numFold = numFold, Info = True, skip=skip, checkAll=checkAll, maximum_Dist=maximum_Dist)
        else:
            f1Best, accBest, aucBest, precBest, recallBest = KNN(normalTrain, normalTest, anomalyTest, BestNeighbor, normalTrainInst, normalTestInst,
                                                  anomalyInst, name, pathWrite, pathMatrix, pathAUC, pathDist, numFold = numFold, Info = False, skip=skip, checkAll=checkAll, maximum_Dist=maximum_Dist)
    
    
    #if neighbors.shape[0] > 1:
    #    printandwrite(pathWrite, "Best Neighbor: " + str(BestNeighbor))
    #    printandwrite(pathWrite, "")
    #    printandwrite(pathWrite, "Overall Best F1 of all KNeighbors: " + str(f1Best))
    #    printandwrite(pathWrite, "Overall Best Accuracy of all KNeighbors: " + str(accBest))
    #    printandwrite(pathWrite, "Overall Best AUC of all KNeighbors: " + str(aucBest))
    #    printandwrite(pathWrite, "========================================================================================")
    #    printandwrite(pathWrite, "========================================================================================")
    #    printandwrite(pathWrite, " ")
    
    
    return f1Best, accBest, aucBest, precBest, recallBest


# Run multiple experiments.
def runExperiment_once(normal, normalTest, anomalyTest, normalTrainInst, normalTestInst, anomalyInst, name, path,
                  neighbors = np.array([10]), numFold = 1, skip = 0.001, show_Info = True):
    
    normalTrain = normal
    normalTest = normalTest
    anomalyTest = anomalyTest
    
    pathWrite = path + name + ".txt"
    pathMatrix = path + "TruthMatrix/"
    pathAUC = path + "AUCGraphs/"
    pathDist = path + "Distances/"
    
    # Check whether the specified path exists or not
    isExist = os.path.exists(path)
    if not isExist:
      # Create a new directory because it does not exist 
      os.makedirs(path)
    
    # Check whether the specified path exists or not
    isExist = os.path.exists(pathMatrix)
    if not isExist:
      # Create a new directory because it does not exist 
      os.makedirs(pathMatrix)
    
    # Check whether the specified path exists or not
    isExist = os.path.exists(pathAUC)
    if not isExist:
      # Create a new directory because it does not exist 
      os.makedirs(pathAUC)
        
    # Check whether the specified path exists or not
    isExist = os.path.exists(pathDist)
    if not isExist:
      # Create a new directory because it does not exist 
      os.makedirs(pathDist)

    if show_Info:
        printandwrite(pathWrite, "Beginning Tests...",True)
    
    BestNeighbor = neighbors[0]
    
    if numFold > 1:
        if show_Info:
            f1Best, accBest, aucBest = KNN_kFold(normalTrain, normalTest, anomalyTest, BestNeighbor, normalTrainInst, normalTestInst,
                                                  anomalyInst, name, pathWrite, pathMatrix, pathAUC, pathDist, numFold = numFold, Info = True, skip=skip)
        else:
            f1Best, accBest, aucBest = KNN_kFold(normalTrain, normalTest, anomalyTest, BestNeighbor, normalTrainInst, normalTestInst,
                                                  anomalyInst, name, pathWrite, pathMatrix, pathAUC, pathDist, numFold = numFold, Info = False, skip=skip)
     
    else:
        if show_Info:
            f1Best, accBest, aucBest = KNN(normalTrain, normalTest, anomalyTest, BestNeighbor, normalTrainInst, normalTestInst,
                                                  anomalyInst, name, pathWrite, pathMatrix, pathAUC, pathDist, numFold = numFold, Info = True, skip=skip)
        else:
            f1Best, accBest, aucBest = KNN(normalTrain, normalTest, anomalyTest, BestNeighbor, normalTrainInst, normalTestInst,
                                                  anomalyInst, name, pathWrite, pathMatrix, pathAUC, pathDist, numFold = numFold, Info = False, skip=skip)
    
    
    #if neighbors.shape[0] > 1:
    #    printandwrite(pathWrite, "Best Neighbor: " + str(BestNeighbor))
    #    printandwrite(pathWrite, "")
    #    printandwrite(pathWrite, "Overall Best F1 of all KNeighbors: " + str(f1Best))
    #    printandwrite(pathWrite, "Overall Best Accuracy of all KNeighbors: " + str(accBest))
    #    printandwrite(pathWrite, "Overall Best AUC of all KNeighbors: " + str(aucBest))
    #    printandwrite(pathWrite, "========================================================================================")
    #    printandwrite(pathWrite, "========================================================================================")
    #    printandwrite(pathWrite, " ")
    
    
    return f1Best, accBest, aucBest

# Run multiple experiments.
def runExperiment_once_AUC(normal, normalTest, anomalyTest, normalTrainInst, normalTestInst, anomalyInst, name, path,
                  neighbors = np.array([10]), numFold = 1, skip = 0.001, show_Info = True):
    
    normalTrain = normal
    normalTest = normalTest
    anomalyTest = anomalyTest
    
    pathWrite = path + name + ".txt"
    pathMatrix = path + "TruthMatrix/"
    pathAUC = path + "AUCGraphs/"
    pathDist = path + "Distances/"
    
    # Check whether the specified path exists or not
    isExist = os.path.exists(path)
    if not isExist:
      # Create a new directory because it does not exist 
      os.makedirs(path)
    
    # Check whether the specified path exists or not
    isExist = os.path.exists(pathMatrix)
    if not isExist:
      # Create a new directory because it does not exist 
      os.makedirs(pathMatrix)
    
    # Check whether the specified path exists or not
    isExist = os.path.exists(pathAUC)
    if not isExist:
      # Create a new directory because it does not exist 
      os.makedirs(pathAUC)
        
    # Check whether the specified path exists or not
    isExist = os.path.exists(pathDist)
    if not isExist:
      # Create a new directory because it does not exist 
      os.makedirs(pathDist)

    if show_Info:
        printandwrite(pathWrite, "Beginning Tests...",True)
    
    BestNeighbor = neighbors[0]
    
    if numFold > 1:
        if show_Info:
            tprs, fprs, aucBest = KNN_kFold(normalTrain, normalTest, anomalyTest, BestNeighbor, normalTrainInst, normalTestInst,
                                                  anomalyInst, name, pathWrite, pathMatrix, pathAUC, pathDist, numFold = numFold, Info = True, skip=skip)
        else:
            tprs, fprs, aucBest = KNN_kFold(normalTrain, normalTest, anomalyTest, BestNeighbor, normalTrainInst, normalTestInst,
                                                  anomalyInst, name, pathWrite, pathMatrix, pathAUC, pathDist, numFold = numFold, Info = False, skip=skip)
     
    else:
        if show_Info:
            tprs, fprs, aucBest = KNN_AUC(normalTrain, normalTest, anomalyTest, BestNeighbor, normalTrainInst, normalTestInst,
                                                  anomalyInst, name, pathWrite, pathMatrix, pathAUC, pathDist, numFold = numFold, Info = True, skip=skip)
        else:
            tprs, fprs, aucBest = KNN_AUC(normalTrain, normalTest, anomalyTest, BestNeighbor, normalTrainInst, normalTestInst,
                                                  anomalyInst, name, pathWrite, pathMatrix, pathAUC, pathDist, numFold = numFold, Info = False, skip=skip)
    
    
    #if neighbors.shape[0] > 1:
    #    printandwrite(pathWrite, "Best Neighbor: " + str(BestNeighbor))
    #    printandwrite(pathWrite, "")
    #    printandwrite(pathWrite, "Overall Best F1 of all KNeighbors: " + str(f1Best))
    #    printandwrite(pathWrite, "Overall Best Accuracy of all KNeighbors: " + str(accBest))
    #    printandwrite(pathWrite, "Overall Best AUC of all KNeighbors: " + str(aucBest))
    #    printandwrite(pathWrite, "========================================================================================")
    #    printandwrite(pathWrite, "========================================================================================")
    #    printandwrite(pathWrite, " ")
    
    
    return tprs, fprs, aucBest


# In[ ]:


# Run multiple experiments.
# Considerts multiple training sets to train one after another.
def runExperiment_Train2(normalTrain1, normalTrain2, normalTest, anomalyTest, normalTrainInst, normalTestInst, anomalyInst, name, path,
                  neighbors = np.array([3,5,10,25,50,75,100]), numFold = 10):

    normalTest = normalTest
    anomalyTest = anomalyTest
    
    pathWrite = path + name + ".txt"
    pathMatrix = path + "TruthMatrix/"
    pathAUC = path + "AUCGraphs/"
    pathDist = path + "Distances/"
    
    # Check whether the specified path exists or not
    isExist = os.path.exists(path)
    if not isExist:
      # Create a new directory because it does not exist 
      os.makedirs(path)
    
    # Check whether the specified path exists or not
    isExist = os.path.exists(pathMatrix)
    if not isExist:
      # Create a new directory because it does not exist 
      os.makedirs(pathMatrix)
    
    # Check whether the specified path exists or not
    isExist = os.path.exists(pathAUC)
    if not isExist:
      # Create a new directory because it does not exist 
      os.makedirs(pathAUC)
    
    f1Bests = []
    accBests = []
    aucBests = []

    printandwrite(pathWrite, "Beginning Tests...",True)
                  
    for neighbor in neighbors:
        #printandwrite(pathWrite, "=======================TEST ON NEIGHBOR = " + str(neighbor) + "=======================")
        
        f1Best, accBest, aucBest = KNN_STROUD_kFold_Train2(normalTrain1, normalTrain2, normalTest, anomalyTest, neighbor, normalTrainInst, normalTestInst,
                                              anomalyInst, name, pathWrite, pathMatrix, pathAUC, pathDist, numFold = numFold, Info = False)
        
        f1Bests.append(f1Best)
        accBests.append(accBest)
        aucBests.append(aucBest)
        
        #printandwrite(pathWrite, " ")
        
    f1Bests = np.array(f1Bests)
    accBests = np.array(accBests)
    aucBests = np.array(aucBests)
    
    outfile = path + 'f1s.npy'
    np.save(outfile, f1Bests)
    
    outfile = path + 'accs.npy'
    np.save(outfile, accBests)
    
    outfile = path + 'aucs.npy'
    np.save(outfile, aucBests)
      
    f1Best = max(f1Bests)
    accBest = max(accBests)
    aucBest = max(aucBests)
    
    accBestFind = 0
    bestIndex = -1
    
    BestNeighbor = findBestNeighbor(accBests, neighbors)
    
    f1Best, accBest, aucBest = KNN_STROUD_kFold_Train2(normalTrain1, normalTrain2, normalTest, anomalyTest, BestNeighbor, normalTrainInst, normalTestInst,
                                              anomalyInst, name, pathWrite, pathMatrix, pathAUC, pathDist, numFold = numFold, Info = True)
        
    
    #if neighbors.shape[0] > 1:
    #    printandwrite(pathWrite, "Best Neighbor: " + str(BestNeighbor))
    #    printandwrite(pathWrite, "")
    #    printandwrite(pathWrite, "Overall Best F1 of all KNeighbors: " + str(f1Best))
    #    printandwrite(pathWrite, "Overall Best Accuracy of all KNeighbors: " + str(accBest))
    #    printandwrite(pathWrite, "Overall Best AUC of all KNeighbors: " + str(aucBest))
    #    printandwrite(pathWrite, "========================================================================================")
    #    printandwrite(pathWrite, "========================================================================================")
    #    printandwrite(pathWrite, " ")
    
    
    return f1Best, accBest, aucBest


# In[ ]:


# Run multiple experiments.
# Considerts multiple training sets to train one after another.
def runExperiment_norm2_anom1(normalTrain1, normalTrain2, normalTest, anomalyTest, normalTrainInst, normalTestInst, anomalyInst, name, path,
                  neighbors = np.array([3,5,10,25,50,75,100]), numFold = 10):

    normalTest = normalTest
    anomalyTest = anomalyTest
    
    pathWrite = path + name + ".txt"
    pathMatrix = path + "TruthMatrix/"
    pathAUC = path + "AUCGraphs/"
    pathDist = path + "Distances/"
    
    # Check whether the specified path exists or not
    isExist = os.path.exists(path)
    if not isExist:
      # Create a new directory because it does not exist 
      os.makedirs(path)
    
    # Check whether the specified path exists or not
    isExist = os.path.exists(pathMatrix)
    if not isExist:
      # Create a new directory because it does not exist 
      os.makedirs(pathMatrix)
    
    # Check whether the specified path exists or not
    isExist = os.path.exists(pathAUC)
    if not isExist:
      # Create a new directory because it does not exist 
      os.makedirs(pathAUC)
    
    f1Bests = []
    accBests = []
    aucBests = []

    printandwrite(pathWrite, "Beginning Tests...",True)
                  
    for neighbor in neighbors:
        #printandwrite(pathWrite, "=======================TEST ON NEIGHBOR = " + str(neighbor) + "=======================")
        
        f1Best, accBest, aucBest = KNN_STROUD_kFold_norm2_anom1(normalTrain1, normalTrain2, normalTest, anomalyTest, neighbor, normalTrainInst, normalTestInst,
                                              anomalyInst, name, pathWrite, pathMatrix, pathAUC, pathDist, numFold = numFold, Info = False)
        
        f1Bests.append(f1Best)
        accBests.append(accBest)
        aucBests.append(aucBest)
        
        #printandwrite(pathWrite, " ")
        
    f1Bests = np.array(f1Bests)
    accBests = np.array(accBests)
    aucBests = np.array(aucBests)
    
    outfile = path + 'f1s.npy'
    np.save(outfile, f1Bests)
    
    outfile = path + 'accs.npy'
    np.save(outfile, accBests)
    
    outfile = path + 'aucs.npy'
    np.save(outfile, aucBests)
      
    f1Best = max(f1Bests)
    accBest = max(accBests)
    aucBest = max(aucBests)
    
    accBestFind = 0
    bestIndex = -1
    
    BestNeighbor = findBestNeighbor(accBests, neighbors)
    
    f1Best, accBest, aucBest = KNN_STROUD_kFold_norm2_anom1(normalTrain1, normalTrain2, normalTest, anomalyTest, BestNeighbor, normalTrainInst, normalTestInst,
                                              anomalyInst, name, pathWrite, pathMatrix, pathAUC, pathDist, numFold = numFold, Info = True)
        
    
    #if neighbors.shape[0] > 1:
    #    printandwrite(pathWrite, "Best Neighbor: " + str(BestNeighbor))
    #    printandwrite(pathWrite, "")
    #    printandwrite(pathWrite, "Overall Best F1 of all KNeighbors: " + str(f1Best))
    #    printandwrite(pathWrite, "Overall Best Accuracy of all KNeighbors: " + str(accBest))
    #    printandwrite(pathWrite, "Overall Best AUC of all KNeighbors: " + str(aucBest))
    #    printandwrite(pathWrite, "========================================================================================")
    #    printandwrite(pathWrite, "========================================================================================")
    #    printandwrite(pathWrite, " ")
    
    
    return f1Best, accBest, aucBest


# In[ ]:


# Run multiple experiments.
# Considerts multiple training sets to train one after another.
def runExperiment_norm2_anom2(normalTrain1, normalTrain2, normalTest, anomalyTest1, anomalyTest2, normalTrainInst, normalTestInst, anomalyInst, name, path,
                  neighbors = np.array([3,5,10,25,50,75,100]), numFold = 10):
    
    pathWrite = path + name + ".txt"
    pathMatrix = path + "TruthMatrix/"
    pathAUC = path + "AUCGraphs/"
    pathDist = path + "Distances/"
    
    # Check whether the specified path exists or not
    isExist = os.path.exists(path)
    if not isExist:
      # Create a new directory because it does not exist 
      os.makedirs(path)
    
    # Check whether the specified path exists or not
    isExist = os.path.exists(pathMatrix)
    if not isExist:
      # Create a new directory because it does not exist 
      os.makedirs(pathMatrix)
    
    # Check whether the specified path exists or not
    isExist = os.path.exists(pathAUC)
    if not isExist:
      # Create a new directory because it does not exist 
      os.makedirs(pathAUC)
    
    f1Bests = []
    accBests = []
    aucBests = []

    printandwrite(pathWrite, "Beginning Tests...",True)
                  
    for neighbor in neighbors:
        #printandwrite(pathWrite, "=======================TEST ON NEIGHBOR = " + str(neighbor) + "=======================")
        
        f1Best, accBest, aucBest = KNN_STROUD_kFold_norm2_anom2(normalTrain1, normalTrain2, normalTest, anomalyTest1, anomalyTest2, neighbor, normalTrainInst, normalTestInst,
                                              anomalyInst, name, pathWrite, pathMatrix, pathAUC, pathDist, numFold = numFold, Info = False)
        
        f1Bests.append(f1Best)
        accBests.append(accBest)
        aucBests.append(aucBest)
        
        #printandwrite(pathWrite, " ")
        
    f1Bests = np.array(f1Bests)
    accBests = np.array(accBests)
    aucBests = np.array(aucBests)
    
    outfile = path + 'f1s.npy'
    np.save(outfile, f1Bests)
    
    outfile = path + 'accs.npy'
    np.save(outfile, accBests)
    
    outfile = path + 'aucs.npy'
    np.save(outfile, aucBests)
      
    f1Best = max(f1Bests)
    accBest = max(accBests)
    aucBest = max(aucBests)
    
    accBestFind = 0
    bestIndex = -1
    
    BestNeighbor = findBestNeighbor(accBests, neighbors)
    
    f1Best, accBest, aucBest = KNN_STROUD_kFold_norm2_anom2(normalTrain1, normalTrain2, normalTest, anomalyTest1, anomalyTest2, BestNeighbor, normalTrainInst, normalTestInst,
                                              anomalyInst, name, pathWrite, pathMatrix, pathAUC, pathDist, numFold = numFold, Info = True)
        
    
    #if neighbors.shape[0] > 1:
    #    printandwrite(pathWrite, "Best Neighbor: " + str(BestNeighbor))
    #    printandwrite(pathWrite, "")
    #    printandwrite(pathWrite, "Overall Best F1 of all KNeighbors: " + str(f1Best))
    #    printandwrite(pathWrite, "Overall Best Accuracy of all KNeighbors: " + str(accBest))
    #    printandwrite(pathWrite, "Overall Best AUC of all KNeighbors: " + str(aucBest))
    #    printandwrite(pathWrite, "========================================================================================")
    #    printandwrite(pathWrite, "========================================================================================")
    #    printandwrite(pathWrite, " ")
    
    
    return f1Best, accBest, aucBest


# In[14]:


# Run multiple experiments.
def runExperimentSVD(normal, normalTest, anomalyTest, normalTrainInst, normalTestInst, anomalyInst, name, path,
                  neighbors = np.array([3,5,10,25,50,75,100]), numFold = 10):
    
    normalTrain = normal
    normalTest = normalTest
    anomalyTest = anomalyTest
    
    pathWrite = path + name + ".txt"
    pathMatrix = path + "TruthMatrix/"
    pathAUC = path + "AUCGraphs/"
    pathDist = path + "Distances/"
    
    f1Bests = []
    accBests = []
    aucBests = []

    printandwrite(pathWrite, "Beginning Tests...",True)
                  
    for neighbor in neighbors:
        #printandwrite(pathWrite, "=======================TEST ON NEIGHBOR = " + str(neighbor) + "=======================")
        
        f1Best, accBest, aucBest = KNN_STROUD_kFoldSVD(normalTrain, normalTest, anomalyTest, neighbor, normalTrainInst, normalTestInst,
                                              anomalyInst, name, pathWrite, pathMatrix, pathAUC, pathDist, numFold = numFold, Info = False)
        
        f1Bests.append(f1Best)
        accBests.append(accBest)
        aucBests.append(aucBest)
        
        #printandwrite(pathWrite, " ")
        
    f1Bests = np.array(f1Bests)
    accBests = np.array(accBests)
    aucBests = np.array(aucBests)
    
    outfile = path + 'f1s.npy'
    np.save(outfile, f1Bests)
    
    outfile = path + 'accs.npy'
    np.save(outfile, accBests)
    
    outfile = path + 'aucs.npy'
    np.save(outfile, aucBests)
      
    f1Best = max(f1Bests)
    accBest = max(accBests)
    aucBest = max(aucBests)
    
    accBestFind = 0
    bestIndex = -1
    
    BestNeighbor = findBestNeighbor(accBests, neighbors)
    
    f1Best, accBest, aucBest = KNN_STROUD_kFoldSVD(normalTrain, normalTest, anomalyTest, BestNeighbor, normalTrainInst, normalTestInst,
                                              anomalyInst, name, pathWrite, pathMatrix, pathAUC, pathDist, numFold = numFold, Info = True)
    
    
    #if neighbors.shape[0] > 1:
    #    printandwrite(pathWrite, "Best Neighbor: " + str(BestNeighbor))
    #    printandwrite(pathWrite, "")
    #    printandwrite(pathWrite, "Overall Best F1 of all KNeighbors: " + str(f1Best))
    #    printandwrite(pathWrite, "Overall Best Accuracy of all KNeighbors: " + str(accBest))
    #    printandwrite(pathWrite, "Overall Best AUC of all KNeighbors: " + str(aucBest))
    #    printandwrite(pathWrite, "========================================================================================")
    #    printandwrite(pathWrite, "========================================================================================")
    #    printandwrite(pathWrite, " ")
    
    
    return f1Best, accBest, aucBest


# In[ ]:


# Run multiple experiments.
def runExperimentVHI(normal, normalTest, anomalyTest, normalTrainInst, normalTestInst, anomalyInst, name, path,
                  neighbors = np.array([3,5,10,25,50,75,100]), numFold = 10):
    
    normalTrain = normal
    normalTest = normalTest
    anomalyTest = anomalyTest
    
    pathWrite = path + name + ".txt"
    pathMatrix = path + "TruthMatrix/"
    pathAUC = path + "AUCGraphs/"
    pathDist = path + "Distances/"
    
    f1Bests = []
    accBests = []
    aucBests = []

    printandwrite(pathWrite, "Beginning Tests...",True)
                  
    for neighbor in neighbors:
        #printandwrite(pathWrite, "=======================TEST ON NEIGHBOR = " + str(neighbor) + "=======================")
        
        f1Best, accBest, aucBest = KNN_STROUD_kFoldVHI(normalTrain, normalTest, anomalyTest, neighbor, normalTrainInst, normalTestInst,
                                              anomalyInst, name, pathWrite, pathMatrix, pathAUC, pathDist, numFold = numFold, Info = False)
        
        f1Bests.append(f1Best)
        accBests.append(accBest)
        aucBests.append(aucBest)
        
        #printandwrite(pathWrite, " ")
        
    f1Bests = np.array(f1Bests)
    accBests = np.array(accBests)
    aucBests = np.array(aucBests)
    
    outfile = path + 'f1s.npy'
    np.save(outfile, f1Bests)
    
    outfile = path + 'accs.npy'
    np.save(outfile, accBests)
    
    outfile = path + 'aucs.npy'
    np.save(outfile, aucBests)
      
    f1Best = max(f1Bests)
    accBest = max(accBests)
    aucBest = max(aucBests)
    
    accBestFind = 0
    bestIndex = -1
    
    BestNeighbor = findBestNeighbor(accBests, neighbors)
    
    f1Best, accBest, aucBest = KNN_STROUD_kFoldVHI(normalTrain, normalTest, anomalyTest, BestNeighbor, normalTrainInst, normalTestInst,
                                              anomalyInst, name, pathWrite, pathMatrix, pathAUC, pathDist, numFold = numFold, Info = True)
    
    
    #if neighbors.shape[0] > 1:
    #    printandwrite(pathWrite, "Best Neighbor: " + str(BestNeighbor))
    #    printandwrite(pathWrite, "")
    #    printandwrite(pathWrite, "Overall Best F1 of all KNeighbors: " + str(f1Best))
    #    printandwrite(pathWrite, "Overall Best Accuracy of all KNeighbors: " + str(accBest))
    #    printandwrite(pathWrite, "Overall Best AUC of all KNeighbors: " + str(aucBest))
    #    printandwrite(pathWrite, "========================================================================================")
    #    printandwrite(pathWrite, "========================================================================================")
    #    printandwrite(pathWrite, " ")
    
    
    return f1Best, accBest, aucBest


# In[13]:


# Run multiple experiments.
def runExperimentSVD_Tran(normal, normalTest, anomalyTest, normalTrainInst, normalTestInst, anomalyInst, name, path,
                          cleanDataset, cut, neighbors = np.array([3,5,10,25,50,75,100]), numFold = 10):
    
    normalTrain = normal
    normalTest = normalTest
    anomalyTest = anomalyTest
    
    pathWrite = path + name + ".txt"
    pathMatrix = path + "TruthMatrix/"
    pathAUC = path + "AUCGraphs/"
    pathDist = path + "Distances/"
    
    f1Bests = []
    accBests = []
    aucBests = []

    printandwrite(pathWrite, "Beginning Tests...",True)
                  
    for neighbor in neighbors:
        #printandwrite(pathWrite, "=======================TEST ON NEIGHBOR = " + str(neighbor) + "=======================")
        
        f1Best, accBest, aucBest = KNN_STROUD_kFoldSVD_Tran(normalTrain, normalTest, anomalyTest, neighbor, normalTrainInst, normalTestInst,
                                              anomalyInst, name, pathWrite, pathMatrix, pathAUC, pathDist, cleanDataset, cut, numFold = numFold, Info = False)
        
        f1Bests.append(f1Best)
        accBests.append(accBest)
        aucBests.append(aucBest)
        
        #printandwrite(pathWrite, " ")
        
    f1Bests = np.array(f1Bests)
    accBests = np.array(accBests)
    aucBests = np.array(aucBests)
    
    outfile = path + 'f1s.npy'
    np.save(outfile, f1Bests)
    
    outfile = path + 'accs.npy'
    np.save(outfile, accBests)
    
    outfile = path + 'aucs.npy'
    np.save(outfile, aucBests)
      
    f1Best = max(f1Bests)
    accBest = max(accBests)
    aucBest = max(aucBests)
    
    accBestFind = 0
    bestIndex = -1
    
    BestNeighbor = findBestNeighbor(accBests, neighbors)
    
    f1Best, accBest, aucBest = KNN_STROUD_kFoldSVD_Tran(normalTrain, normalTest, anomalyTest, BestNeighbor, normalTrainInst, normalTestInst,
                                              anomalyInst, name, pathWrite, pathMatrix, pathAUC, pathDist, cleanDataset, cut, numFold = numFold, Info = True)
    
    
    #if neighbors.shape[0] > 1:
    #    printandwrite(pathWrite, "Best Neighbor: " + str(BestNeighbor))
    #    printandwrite(pathWrite, "")
    #    printandwrite(pathWrite, "Overall Best F1 of all KNeighbors: " + str(f1Best))
    #    printandwrite(pathWrite, "Overall Best Accuracy of all KNeighbors: " + str(accBest))
    #    printandwrite(pathWrite, "Overall Best AUC of all KNeighbors: " + str(aucBest))
    #    printandwrite(pathWrite, "========================================================================================")
    #    printandwrite(pathWrite, "========================================================================================")
    #    printandwrite(pathWrite, " ")
    
    
    return f1Best, accBest, aucBest


# In[15]:


# Run multiple experiments.
def runExperimentVHI_Tran(normal, normalTest, anomalyTest, normalTrainInst, normalTestInst, anomalyInst, name, path, 
                          cleanDataset, cut, neighbors = np.array([3,5,10,25,50,75,100]), numFold = 10):
    
    normalTrain = normal
    normalTest = normalTest
    anomalyTest = anomalyTest
    
    pathWrite = path + name + ".txt"
    pathMatrix = path + "TruthMatrix/"
    pathAUC = path + "AUCGraphs/"
    pathDist = path + "Distances/"
    
    f1Bests = []
    accBests = []
    aucBests = []

    printandwrite(pathWrite, "Beginning Tests...",True)
                  
    for neighbor in neighbors:
        #printandwrite(pathWrite, "=======================TEST ON NEIGHBOR = " + str(neighbor) + "=======================")
        
        f1Best, accBest, aucBest = KNN_STROUD_kFoldVHI_Tran(normalTrain, normalTest, anomalyTest, neighbor, normalTrainInst, normalTestInst,
                                              anomalyInst, name, pathWrite, pathMatrix, pathAUC, pathDist, cleanDataset, cut, numFold = numFold, Info = False)
        
        f1Bests.append(f1Best)
        accBests.append(accBest)
        aucBests.append(aucBest)
        
        #printandwrite(pathWrite, " ")
        
    f1Bests = np.array(f1Bests)
    accBests = np.array(accBests)
    aucBests = np.array(aucBests)
    
    outfile = path + 'f1s.npy'
    np.save(outfile, f1Bests)
    
    outfile = path + 'accs.npy'
    np.save(outfile, accBests)
    
    outfile = path + 'aucs.npy'
    np.save(outfile, aucBests)
      
    f1Best = max(f1Bests)
    accBest = max(accBests)
    aucBest = max(aucBests)
    
    accBestFind = 0
    bestIndex = -1
    
    BestNeighbor = findBestNeighbor(accBests, neighbors)
    
    f1Best, accBest, aucBest = KNN_STROUD_kFoldVHI_Tran(normalTrain, normalTest, anomalyTest, BestNeighbor, normalTrainInst, normalTestInst,
                                              anomalyInst, name, pathWrite, pathMatrix, pathAUC, pathDist, cleanDataset, cut, numFold = numFold, Info = True)
    
    
    #if neighbors.shape[0] > 1:
    #    printandwrite(pathWrite, "Best Neighbor: " + str(BestNeighbor))
    #    printandwrite(pathWrite, "")
    #    printandwrite(pathWrite, "Overall Best F1 of all KNeighbors: " + str(f1Best))
    #    printandwrite(pathWrite, "Overall Best Accuracy of all KNeighbors: " + str(accBest))
    #    printandwrite(pathWrite, "Overall Best AUC of all KNeighbors: " + str(aucBest))
    #    printandwrite(pathWrite, "========================================================================================")
    #    printandwrite(pathWrite, "========================================================================================")
    #    printandwrite(pathWrite, " ")
    
    
    return f1Best, accBest, aucBest

