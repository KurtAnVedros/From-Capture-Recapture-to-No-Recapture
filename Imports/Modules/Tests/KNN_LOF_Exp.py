# imports
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
import scipy.io as sio
import csv
import os
from Algorithms import SVDDenoise
from scipy.signal.signaltools import wiener
from sklearn.neighbors import KNeighborsRegressor
import warnings
from scipy.signal import savgol_filter
from scipy.spatial import distance
from sklearn.neighbors import NearestNeighbors
import keras
from skimage.restoration import denoise_wavelet

# Changes size of plots
plt.rcParams['figure.figsize'] = [15, 5]


params = {'legend.fontsize': 50,
          'figure.figsize': (45, 15),
         'axes.labelsize': 60,
         'axes.titlesize':70,
         'xtick.labelsize':50,
         'ytick.labelsize':50,
          'lines.linewidth':10,
         'lines.markersize':10}
plt.rcParams.update(params)


# Save information

## DESCRIPTION: Saves the basic information on given Experiment info into separate csv files
## tns = true negatives
## fps = false positives
## tps = true positives
## fprs = false positive rate
## tprs = true positive rate
## location = file location to save the data
## preName = name of the data.
def saveToFileSplit(tns, fps, fns, tps, fprs, tprs, location, preName):
    # try:
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


# Save f1s, accs, auc of each fold.
def saveScores(f1s, accs, auc, location, preName):
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

# Preprocess methods
def autoencoder(signals, model):
    denoised = np.copy(signals)
    for index in range(signals.shape[0]):
        #denoised[index] = np.squeeze(model.predict(np.reshape(signals[index], (1,1, -1))))
        denoised[index] = np.squeeze(model.predict(np.reshape(signals[index], (1, signals[index].shape[0], 1))))
    return denoised


def KNN_Regressor(signals, neighbor):
    clf = KNeighborsRegressor(n_neighbors=neighbor, weights='uniform')
    timeVector = np.arange(0, signals.shape[1], 1)
    timeVector = np.expand_dims(timeVector, axis=1)

    denoised = np.zeros((signals.shape[0], signals.shape[1]))
    for index in range(signals.shape[0]):
        clf.fit(timeVector, signals[index])
        denoised[index] = clf.predict(timeVector)

    return denoised

def wavelet(signals, Level, Wavelet):
    denoised = np.zeros((signals.shape[0], signals.shape[1]))
    for index in range(signals.shape[0]):
        denoised[index] = denoise_wavelet(signals[index], method='BayesShrink', mode = 'soft',
                                          wavelet_levels = Level, wavelet = Wavelet, rescale_sigma = 'true')
    return denoised

import os
def checkAndMakeFolder(path):
    # Check whether the specified path exists or not
    isExist = os.path.exists(path)
    if not isExist:
       # Create a new directory because it does not exist
       os.makedirs(path)

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


# Experiment model.
## expType          = the type of experiment to run on given in number selection format
##                  1 = KNN, 2 = LOF, 3 = KNN STROUD, 4 = LOF STROUD
## preprocess       = the type of preprocessing to run before the experiment given in number selection format
##                  0 = None, 1 = Weiner, 2 = Sta-Gol, 3 = KNN Regressor, 4 = autoencoder, 5 = SVD, 6 = Vhi,
##                  7 = SVD Transfer, 8 = Vhi Transfer, 9 = Wavelet
## ppParameter_1_Train    = preprocessing parameter 1. this is the first parameter used to perform the preprocessing.
##                  Weiner = power, StaGol = windows, KNNReg = neighbors, autoencoder = model
##                  SVD/Vhi/SVD Transfer/ Vhi Transfer = cutting point, Wavelet = Level
## ppParameter_2_Train    = preprocessing parameter 2. this is the second parameter used to perform the preprocessing.
##                  Weiner = N/A, StaGol = Polynomial, KNNReg = N/A, autoencoder = N/A,
##                  SVD/Vhi = N/A, SVD Transfer/Vhi Transfer = Clean Dataset, Wavelet = Wavelet formula
## ppParameter_1_test    = preprocessing parameter 1. this is the first parameter used to perform the preprocessing.
##                  Weiner = power, StaGol = windows, KNNReg = neighbors, autoencoder = model
##                  SVD/Vhi/SVD Transfer/ Vhi Transfer = cutting point, Wavelet = Level
## ppParameter_2_test    = preprocessing parameter 2. this is the second parameter used to perform the preprocessing.
##                  Weiner = N/A, StaGol = Polynomial, KNNReg = N/A, autoencoder = N/A,
##                  SVD/Vhi = N/A, SVD Transfer/Vhi Transfer = Clean Dataset, Wavelet = Wavelet formula
## normalTrain      = normal dataset to train on.
## normalTest       = normal dataset to test on.
## anomalyTest      = the anomalous dataset to test on.
## neighbors        = the numvber of neighbors to consider for KNN or LOF experiment.
## normalTrainInst  = the string name of the normal dataset to train on.
## normalTestInst   = the string name of the normal dataset to test on.
## anomalyInst      = the string name of the anomalous dataset to test on.
## pathWrite        = the file location to save the txt file of the best results.
## pathMatrix       = the file location to save the matrix information in csv files.
## pathAUC          = the file location to save the AUC graph
## numFold          = the number of folds to perform kFold test.
## Info             = True/False on showing the information for the experiment.
## randomseed       = randomseed setting to produce same results.
## lengthTest       = length of normal and abnormal Test sets
def experiment(expType, preprocess, ppParameter_1_Train, ppParameter_1_Test, ppParameter_2_Train, ppParameter_2_Test,
               normalTrain, normalTest, anomalyTest, neighbors, normalTrainInst, normalTestInst, anomalyInst,
                     pathWrite, pathMatrix, pathAUC, pathDist, numFold=10, Info=True, randomseed=25, lengthTest = -1):
    
    # sets the random seed
    random.seed(randomseed)
    np.random.seed(randomseed)
    tf.random.set_seed(randomseed)

    # lists for holding data
    aucs = []
    f1s = []
    accs = []
    
    fprsies = []
    tprsies = []
    tprs_Mean = []

    count = 0

    methodName = ""

    # initialise and specific the number of splits
    kf = KFold(n_splits=numFold)
    # use the normalTrain to select indexes for training and testing subsets
    kf.get_n_splits(normalTrain)
    # for each split
    for train_index, test_index in kf.split(normalTrain):

        # Section for obtaining the folds training and testing set
        if lengthTest > 0:
            test_index = test_index[0:lengthTest]

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


        # Section for doing preprocessing on the training and testing set
        ## None
        if preprocess == 0:
            transformed_training_set = X_train
            transformed_test_set = X_test
        ## Weiner
        elif preprocess == 1:
            transformed_training_set = wiener(X_train, noise = ppParameter_1_Train)
            transformed_test_set = wiener(X_test, noise = ppParameter_1_Test)
        ## Sta-Gol
        elif preprocess == 2:
            transformed_training_set = savgol_filter(X_train, ppParameter_1_Train, ppParameter_2_Train)
            transformed_test_set = savgol_filter(X_test, ppParameter_1_Test, ppParameter_2_Test)
        ## KNN Regressor
        elif preprocess == 3:
            transformed_training_set = KNN_Regressor(X_train, ppParameter_1_Train)
            transformed_test_set = KNN_Regressor(X_test, ppParameter_1_Test)
        ## Autoencoder
        elif preprocess == 4:
            transformed_training_set = autoencoder(X_train, ppParameter_1_Train)
            transformed_test_set = autoencoder(X_test, ppParameter_1_Test)
        ## SVD
        elif preprocess == 5:
            s_clean_train = SVDDenoise.DeduceSigmaFilter(X_train, ppParameter_1_Train)
            transformed_training_set = SVDDenoise.ApplySigmaFilter(X_train, s_clean_train)
            s_clean_test = SVDDenoise.DeduceSigmaFilter(X_test, ppParameter_1_Test)
            transformed_test_set = SVDDenoise.ApplySigmaFilter(X_test, s_clean_test)

            # NOTE should not concatenate
            #length = X_train.shape[0]
            #combined = np.concatenate((X_train, X_test), axis = 0)
            #s_clean_test = SVDDenoise.DeduceSigmaFilter(combined, ppParameter_1)
            #combinedDenoised = SVDDenoise.ApplySigmaFilter(combined, s_clean_test)
            #transformed_test_set = combinedDenoised[length:]

        ## Vhi
        elif preprocess == 6:
            v_new_train = SVDDenoise.DeduceVhiFilter(X_train, ppParameter_1_Train)
            transformed_training_set = SVDDenoise.ApplyVhiFilter(X_train, v_new_train)
            v_new_test = SVDDenoise.DeduceVhiFilter(X_test, ppParameter_1_Test)
            transformed_test_set = SVDDenoise.ApplyVhiFilter(X_test, v_new_test)
        ## SVD Transfer
        elif preprocess == 7:
            s_clean = SVDDenoise.DeduceSigmaFilter(ppParameter_2_Train, ppParameter_1_Train)
            transformed_training_set = SVDDenoise.ApplySigmaFilter(X_train, s_clean)
            transformed_test_set = SVDDenoise.ApplySigmaFilter(X_test, s_clean)
        ## Vhi Transfer
        elif preprocess == 8:
            v_new_train = SVDDenoise.DeduceVhiFilter(ppParameter_2_Train, ppParameter_1_Train)
            transformed_training_set = SVDDenoise.ApplyVhiFilter(X_train, v_new)
            transformed_test_set = SVDDenoise.ApplyVhiFilter(X_test, v_new)
        ## Wavelet
        elif preprocess == 9:
            transformed_training_set = wavelet(X_train, ppParameter_1_Train, ppParameter_2_Train)
            transformed_test_set = wavelet(X_test, ppParameter_1_Test, ppParameter_2_Test)


        # Section for doing experiment type
        ## KNN
        if expType == 1:
            nbrs = NearestNeighbors(n_neighbors=neighbors)
            nbrs.fit(transformed_training_set)

            distTrain, indexTrain = nbrs.kneighbors(transformed_training_set)
            distTest, indexTest = nbrs.kneighbors(transformed_test_set)

            mini = np.amin(distTrain)
            maxi = np.amax(distTrain)

            outfile = pathDist + 'distTrain_N=' + str(neighbors) + '_Fold=' + str(count) + '.npy'
            np.save(outfile, distTrain)

            outfile = pathDist + 'distTest_N=' + str(neighbors) + '_Fold=' + str(count) + '.npy'
            np.save(outfile, distTest)

            X_Test_KNNScores = distTest / maxi

            # run threshold test and obtain information
            list_of_metrics = getEvaluationKNN(X_Test_KNNScores, y_test, 1000)

            methodName = "KNN"

        ## LOF (not yet implemented)
        #elif expType == 2:

        ## KNN-STOUD
        elif expType == 3:
            # compute the strangess scores: the base and test KNN score
            # method
            base_strangeness_list = KNN.executeKNN(transformed_training_set, transformed_training_set, neighbors)
            test_strangeness_list = KNN.executeKNN(transformed_training_set, transformed_test_set, neighbors)

            # compute the p_values
            # Uses the STROUD method
            p_values = STROUD.executeSTROUD(base_strangeness_list, test_strangeness_list, "false")

            # compute the metrics (fpr, tpr, auc) for each confidence level
            list_of_metrics = ConfidenceEstimation.getEvaluation(p_values, y_test, 0, 1, 0.001)

            methodName = "KNN-STROUD"

        ## LOF-STROUD
        elif expType == 4:
            # compute the strangess scores: the base and test LOF score
            # method
            base_strangeness_list, test_strangeness_list = LOF.executeLOF(transformed_training_set,
                                                                          transformed_test_set, neighbors)

            # compute the p_values
            # Uses the STROUD method
            p_values = STROUD.executeSTROUD(base_strangeness_list, test_strangeness_list, "false")

            # compute the metrics (fpr, tpr, auc) for each confidence level
            list_of_metrics = ConfidenceEstimation.getEvaluation(p_values, y_test, 0, 1, 0.001)

            methodName = "LOF-STROUD"

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

        # Save the tns, fps, fns, tps matrix.
        saveToFileSplit(tnsSec, fpsSec, fnsSec, tpsSec, fprs, tprs, pathMatrix,
                        "n=" + str(neighbors) + "_fold" + str(count))

        # Save f1s, accs, auc of each fold.
        saveScores(f1sSec, accsSec, np.array([auc_Score]), pathMatrix, "n=" + str(neighbors) + "_fold" + str(count))

        # Use this to see AUC graphs
        rocTitle = anomalyInst
        aucScoreSave = int(auc_Score * 1000)
        if Info:
            ROC.plotROC(fprs, tprs, "red", 10, "ROC", "FP Rate", "TP Rate", rocTitle, 4, auc_Score)
            ROC.plotROC(fprs, tprs, "red", 10, "ROC", "FP Rate", "TP Rate", rocTitle, 4, auc_Score, pathAUC + "n=" + str(neighbors) + "_fold" + str(count) + "AUC=" + str(aucScoreSave))

        count = count + 1
        
    if Info:
        ROC.plotROCMultiple(fprsies, tprsies, aucs, tprs_Mean, "green", 10, "FP Rate", "TP Rate", rocTitle, 4)
        ROC.plotROCMultiple(fprsies, tprsies, aucs, tprs_Mean, "green", 10, "FP Rate", "TP Rate", rocTitle, 4, pathAUC +"_Full")

    if Info:
        printandwrite(pathWrite, "Method: " + methodName)
        printandwrite(pathWrite, "Normal Dataset Used: " + normalTrainInst)
        printandwrite(pathWrite, "Size of Training Set: " + str(X_train_size))
        printandwrite(pathWrite,
                      "Test Set Used: " + normalTestInst + " normal along with " + anomalyInst + " anomaly instances")
        printandwrite(pathWrite,
                      "Size of Test Set: " + str(X_test_A_size) + " (normal) " + str(X_test_B_size) + " (anomalous) ")
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

    return np.mean(f1s), np.mean(accs), np.mean(aucs)

# finds the best neighbor given the results
def findBestNeighbor(results, neighbors):
    Best = 0
    bestIndex = -1

    for index in range(neighbors.shape[0]):
        if results[index] > Best:
            bestIndex = index
            Best = results[index]

    return neighbors[bestIndex]

# writes
def printandwrite(loc, text, firstLine = False):
    print(text)
    if firstLine:
        sourceFile = open(loc, 'w')
    else:
        sourceFile = open(loc, 'a')
    print(text, file = sourceFile)
    sourceFile.close()

# Run multiple experiments.
## 1. expTypeS = the type of experiment to run on given in string format
##                  KNN, LOF, KNN STROUD, LOF STROUD
## 2. preprocesS       = the type of preprocessing to run before the experiment given in string format
##                  None, Weiner, StaGol, KNNRegressor, autoencoder, SVD, Vhi,
##                  SVDTransfer, VhiTransfer, wavelet
## 3. ppParameter_1_Range    = preprocessing parameter 1. this is the first parameter used to perform the preprocessing.
##                  Weiner = power, StaGol = windows, KNNReg = neighbors, autoencoder = model
##                  SVD/Vhi/SVD Transfer/ Vhi Transfer = cutting point, Wavelet = Level
## 4. ppParameter_2_Range   = preprocessing parameter 2. this is the second parameter used to perform the preprocessing.
##                  Weiner = N/A, StaGol = Polynomial, KNNReg = N/A, autoencoder = N/A,
##                  SVD/Vhi = N/A, SVD Transfer/Vhi Transfer = Clean Dataset, Wavelet = Wavelet formula
## 5. lookForBest      = sting of the best result to look for. (either ACC, F1, or AUC)
## 6. normalTrain      = the normal dataset to train on.
## 7. normalTest       = the normal dataset to test on.
## 8. anomalyTest      = the anomalous dataset to test on.
## 9. normalTrainInst  = the string name of the normal dataset to train on.
## 10. normalTestInst   = the string name of the normal dataset to test on.
## 11. anomalyInst      = the string name of the anomalous dataset to test on.
## 12. name             = the general name for the experiment.
## 13. path             = folder location to save the data.
## 14. neighbors        = the different number of neighbors to test given in a numpy 1D array.
## 15. numFolds         = the number of Folds to perform kFold.
## 16. lengthTest       = the total number of samples used in the test set. Must be an even number.
## 17. randSeed         = Random seed variable to reproduce results if needed.
## 18. sameRange        = Whether to use the same parameters for training and testing set. Standard set to false
## 19. ppParameter_1_Range_TestOnly = range for specifially test set only for preprocessing parameter 1.
##                                      Leave at -1 if same range as ppParameter_1_Range.
## 20. ppParameter_2_Range_TestOnly = range for specifially test set only for preprocessing parameter 2.
##                                      Leave at -1 if same range as ppParameter_2_Range.
def runExperiment(expTypeS, preprocessS, ppParameter_1_Range, ppParameter_2_Range,
                  lookForBest, normalTrain, normalTest,
                  anomalyTest, normalTrainInst, normalTestInst, anomalyInst, name, path,
                  neighbors=np.array([3, 5, 10, 25, 50, 75, 100]), numFold=10, lengthTest = -1, randSeed=25, sameRange = False,
                  ppParameter_1_Range_TestOnly = np.array([-1]), ppParameter_2_Range_TestOnly = np.array([-1])):

    normalTrain = normalTrain
    normalTest = normalTest
    anomalyTest = anomalyTest

    # For file directory reasons
    pathWrite = path + name + ".txt"
    pathMatrix = path + "TruthMatrix/"
    pathAUC = path + "AUCGraphs/"
    pathDist = path + "Distances/"
    
    checkAndMakeFolder(path)
    checkAndMakeFolder(pathMatrix)
    checkAndMakeFolder(pathAUC)
    checkAndMakeFolder(pathDist)

    # List of performance measures
    f1Bests = []
    accBests = []
    aucBests = []

    # Lower case string input
    expTypeS = expTypeS.lower()
    preprocessS = preprocessS.lower()
    lookForBest = lookForBest.lower()

    expType = -1
    preprocess = -1

    # For numbericalizing method selection
    if expTypeS == "knn":
        expType = 1
    elif expTypeS == "lof":
        expType = 2
    elif expTypeS == "knn-stroud":
        expType = 3
    elif expTypeS == "lof-stroud":
        expType = 4

    # For numbericalizing filter selection
    if preprocessS == "none":
        preprocess = 0
    elif preprocessS == "weiner":
        preprocess = 1
    elif preprocessS == "stagol":
        preprocess = 2
    elif preprocessS == "knnregressor":
        preprocess = 3
    elif preprocessS == "autoencoder":
        preprocess = 4
    elif preprocessS == "svd":
        preprocess = 5
    elif preprocessS == "vhi":
        preprocess = 6
    elif preprocessS == "svdtransfer":
        preprocess = 7
    elif preprocessS == "vhitransfer":
        preprocess = 8
    elif preprocessS == "wavelet":
        preprocess = 9

    printandwrite(pathWrite, "Beginning Tests...", True)

    # for timing the experiment run time.
    start = timeit.default_timer()

    bestResult = 0
    bestNeighbor = neighbors[0]
    bestTrainCP = 0
    bestTestCP = 0

    for neighbor in neighbors:
        # Do brute force search for best cp.
        if sameRange == False:
            if preprocess == 2:
                for train_1 in ppParameter_1_Range:
                    for test_1 in ppParameter_1_Range:
                        for train_2 in ppParameter_2_Range:
                            for test_2 in ppParameter_2_Range:
                                # polynomial (2) must be less than window(1)

                                if train_2 < train_1:
                                    if test_2 < test_1:
                                        f1Best, accBest, aucBest = experiment(expType, preprocess, train_1, test_1, train_2, test_2,
                                                                              normalTrain, normalTest,
                                                                              anomalyTest, neighbor, normalTrainInst, normalTestInst,
                                                                              anomalyInst,
                                                                              pathWrite, pathMatrix, pathAUC, pathDist, numFold=numFold,
                                                                              Info=False, lengthTest = lengthTest,
                                                                              randomseed=randSeed)

                                        if lookForBest == 'acc':
                                            if bestResult < accBest:
                                                bestResult = accBest
                                                bestTrain_1 = train_1
                                                bestTest_1 = test_1
                                                bestTrain_2 = train_2
                                                bestTest_2 = test_2
                                                bestNeighbor = neighbor

                                        elif lookForBest == 'f1':
                                            if bestResult < f1Best:
                                                bestResult = f1Best
                                                bestTrain_1 = train_1
                                                bestTest_1 = test_1
                                                bestTrain_2 = train_2
                                                bestTest_2 = test_2
                                                bestNeighbor = neighbor

                                        elif lookForBest == 'auc':
                                            if bestResult < aucBest:
                                                bestResult = aucBest
                                                bestTrain_1 = train_1
                                                bestTest_1 = test_1
                                                bestTrain_2 = train_2
                                                bestTest_2 = test_2
                                                bestNeighbor = neighbor

            elif preprocess == 4:
                f1Best, accBest, aucBest = experiment(expType, preprocess, ppParameter_1_Range, ppParameter_2_Range, -1, -1,
                                                      normalTrain, normalTest,
                                                      anomalyTest, neighbor, normalTrainInst, normalTestInst,
                                                      anomalyInst,
                                                      pathWrite, pathMatrix, pathAUC, pathDist, numFold=numFold,
                                                      Info=False, lengthTest=lengthTest,
                                                      randomseed=randSeed)

                if lookForBest == 'acc':
                    if bestResult < accBest:
                        bestResult = accBest
                        bestNeighbor = neighbor

                elif lookForBest == 'f1':
                    if bestResult < f1Best:
                        bestResult = f1Best
                        bestNeighbor = neighbor

                elif lookForBest == 'auc':
                    if bestResult < aucBest:
                        bestResult = aucBest
                        bestNeighbor = neighbor

            elif preprocess == 9:
                for train_1 in ppParameter_1_Range:
                    for test_1 in ppParameter_1_Range:
                        for train_2 in ppParameter_2_Range:
                            for test_2 in ppParameter_2_Range:
                                f1Best, accBest, aucBest = experiment(expType, preprocess, train_1, test_1, train_2, test_2,
                                                                      normalTrain, normalTest,
                                                                      anomalyTest, neighbor, normalTrainInst, normalTestInst,
                                                                      anomalyInst,
                                                                      pathWrite, pathMatrix, pathAUC, pathDist, numFold=numFold,
                                                                      Info=False, lengthTest = lengthTest,
                                                                      randomseed=randSeed)

                                if lookForBest == 'acc':
                                    if bestResult < accBest:
                                        bestResult = accBest
                                        bestTrain_1 = train_1
                                        bestTest_1 = test_1
                                        bestTrain_2 = train_2
                                        bestTest_2 = test_2
                                        bestNeighbor = neighbor

                                elif lookForBest == 'f1':
                                    if bestResult < f1Best:
                                        bestResult = f1Best
                                        bestTrain_1 = train_1
                                        bestTest_1 = test_1
                                        bestTrain_2 = train_2
                                        bestTest_2 = test_2
                                        bestNeighbor = neighbor

                                elif lookForBest == 'auc':
                                    if bestResult < aucBest:
                                        bestResult = aucBest
                                        bestTrain_1 = train_1
                                        bestTest_1 = test_1
                                        bestTrain_2 = train_2
                                        bestTest_2 = test_2
                                        bestNeighbor = neighbor

            else:
                if np.all(ppParameter_1_Range_TestOnly == -1):
                    ppParameter_1_Range_TestOnly = ppParameter_1_Range
                for train_1 in ppParameter_1_Range:
                    for test_1 in ppParameter_1_Range_TestOnly:
                        f1Best, accBest, aucBest = experiment(expType, preprocess, train_1, test_1, -1, -1,
                                                              normalTrain, normalTest,
                                                              anomalyTest, neighbor, normalTrainInst, normalTestInst,
                                                              anomalyInst,
                                                              pathWrite, pathMatrix, pathAUC, pathDist, numFold=numFold,
                                                              Info=False, lengthTest = lengthTest,
                                                              randomseed=randSeed)

                        if lookForBest == 'acc':
                            if bestResult < accBest:
                                bestResult = accBest
                                bestTrain_1 = train_1
                                bestTest_1 = test_1
                                bestNeighbor = neighbor

                        elif lookForBest == 'f1':
                            if bestResult < f1Best:
                                bestResult = f1Best
                                bestTrain_1 = train_1
                                bestTest_1 = test_1
                                bestNeighbor = neighbor

                        elif lookForBest == 'auc':
                            if bestResult < aucBest:
                                bestResult = aucBest
                                bestTrain_1 = train_1
                                bestTest_1 = test_1
                                bestNeighbor = neighbor


        else:
            if preprocess == 2:
                for train_1 in ppParameter_1_Range:
                    for train_2 in ppParameter_2_Range:
                        # polynomial (2) must be less than window(1)
                        if train_2 < train_1:
                            f1Best, accBest, aucBest = experiment(expType, preprocess, train_1, train_1,
                                                                  train_2, train_2,
                                                                  normalTrain, normalTest,
                                                                  anomalyTest, neighbor, normalTrainInst,
                                                                  normalTestInst,
                                                                  anomalyInst,
                                                                  pathWrite, pathMatrix, pathAUC, pathDist,
                                                                  numFold=numFold,
                                                                  Info=False, lengthTest=lengthTest,
                                                                  randomseed=randSeed)

                            if lookForBest == 'acc':
                                if bestResult < accBest:
                                    bestResult = accBest
                                    bestTrain_1 = train_1
                                    bestTest_1 = train_1
                                    bestTrain_2 = train_2
                                    bestTest_2 = train_2
                                    bestNeighbor = neighbor

                            elif lookForBest == 'f1':
                                if bestResult < f1Best:
                                    bestResult = f1Best
                                    bestTrain_1 = train_1
                                    bestTest_1 = train_1
                                    bestTrain_2 = train_2
                                    bestTest_2 = train_2
                                    bestNeighbor = neighbor

                            elif lookForBest == 'auc':
                                if bestResult < aucBest:
                                    bestResult = aucBest
                                    bestTrain_1 = train_1
                                    bestTest_1 = train_1
                                    bestTrain_2 = train_2
                                    bestTest_2 = train_2
                                    bestNeighbor = neighbor

            elif preprocess == 4:
                f1Best, accBest, aucBest = experiment(expType, preprocess, ppParameter_1_Range, ppParameter_1_Range, -1,
                                                      -1,
                                                      normalTrain, normalTest,
                                                      anomalyTest, neighbor, normalTrainInst, normalTestInst,
                                                      anomalyInst,
                                                      pathWrite, pathMatrix, pathAUC, pathDist, numFold=numFold,
                                                      Info=False, lengthTest=lengthTest,
                                                      randomseed=randSeed)

                if lookForBest == 'acc':
                    if bestResult < accBest:
                        bestResult = accBest
                        bestNeighbor = neighbor

                elif lookForBest == 'f1':
                    if bestResult < f1Best:
                        bestResult = f1Best
                        bestNeighbor = neighbor

                elif lookForBest == 'auc':
                    if bestResult < aucBest:
                        bestResult = aucBest
                        bestNeighbor = neighbor

            elif preprocess == 9:
                for train_1 in ppParameter_1_Range:
                    for train_2 in ppParameter_2_Range:
                        f1Best, accBest, aucBest = experiment(expType, preprocess, train_1, train_1, train_2,
                                                              train_2,
                                                              normalTrain, normalTest,
                                                              anomalyTest, neighbor, normalTrainInst,
                                                              normalTestInst,
                                                              anomalyInst,
                                                              pathWrite, pathMatrix, pathAUC, pathDist,
                                                              numFold=numFold,
                                                              Info=False, lengthTest=lengthTest,
                                                              randomseed=randSeed)

                        if lookForBest == 'acc':
                            if bestResult < accBest:
                                bestResult = accBest
                                bestTrain_1 = train_1
                                bestTest_1 = train_1
                                bestTrain_2 = train_2
                                bestTest_2 = train_2
                                bestNeighbor = neighbor

                        elif lookForBest == 'f1':
                            if bestResult < f1Best:
                                bestResult = f1Best
                                bestTrain_1 = train_1
                                bestTest_1 = train_1
                                bestTrain_2 = train_2
                                bestTest_2 = train_2
                                bestNeighbor = neighbor

                        elif lookForBest == 'auc':
                            if bestResult < aucBest:
                                bestResult = aucBest
                                bestTrain_1 = train_1
                                bestTest_1 = train_1
                                bestTrain_2 = train_2
                                bestTest_2 = train_2
                                bestNeighbor = neighbor

            else:
                for train_1 in ppParameter_1_Range:
                    f1Best, accBest, aucBest = experiment(expType, preprocess, train_1, train_1, -1, -1,
                                                          normalTrain, normalTest,
                                                          anomalyTest, neighbor, normalTrainInst, normalTestInst,
                                                          anomalyInst,
                                                          pathWrite, pathMatrix, pathAUC, pathDist, numFold=numFold,
                                                          Info=False, lengthTest=lengthTest,
                                                          randomseed=randSeed)

                    if lookForBest == 'acc':
                        if bestResult < accBest:
                            bestResult = accBest
                            bestTrain_1 = train_1
                            bestTest_1 = train_1
                            bestNeighbor = neighbor

                    elif lookForBest == 'f1':
                        if bestResult < f1Best:
                            bestResult = f1Best
                            bestTrain_1 = train_1
                            bestTest_1 = train_1
                            bestNeighbor = neighbor

                    elif lookForBest == 'auc':
                        if bestResult < aucBest:
                            bestResult = aucBest
                            bestTrain_1 = train_1
                            bestTest_1 = train_1
                            bestNeighbor = neighbor

    # for timing the experiment run time.
    startSec = timeit.default_timer()
    # Weiner
    if preprocess == 1:
        printandwrite(pathWrite, "Best Neighbor found: " + str(bestNeighbor))
        printandwrite(pathWrite, "Best Train Power found: " + str(bestTrain_1))
        printandwrite(pathWrite, "Best Test Power found: " + str(bestTest_1))
        f1Best, accBest, aucBest = experiment(expType, preprocess, bestTrain_1, bestTest_1, -1, -1, normalTrain,
                                              normalTest,
                                              anomalyTest, bestNeighbor, normalTrainInst, normalTestInst, anomalyInst,
                                              pathWrite, pathMatrix, pathAUC, pathDist, numFold=numFold, Info=True,
                                              randomseed=randSeed, lengthTest=lengthTest)

    # StaGol
    elif preprocess == 2:
        printandwrite(pathWrite, "Best Neighbor found: " + str(bestNeighbor))
        printandwrite(pathWrite, "Best Train Window found: " + str(bestTrain_1))
        printandwrite(pathWrite, "Best Test Window found: " + str(bestTest_1))
        printandwrite(pathWrite, "Best Train Polynomial found: " + str(bestTrain_2))
        printandwrite(pathWrite, "Best Test Polynomial found: " + str(bestTest_2))
        f1Best, accBest, aucBest = experiment(expType, preprocess, bestTrain_1, bestTest_1, bestTrain_2, bestTest_2, normalTrain,
                                              normalTest,
                                              anomalyTest, bestNeighbor, normalTrainInst, normalTestInst, anomalyInst,
                                              pathWrite, pathMatrix, pathAUC, pathDist, numFold=numFold, Info=True,
                                              randomseed=randSeed, lengthTest =lengthTest)

    # KNNReg
    elif preprocess == 3:
        printandwrite(pathWrite, "Best Neighbor found: " + str(bestNeighbor))
        printandwrite(pathWrite, "Best Train KNNReg Neighbor found: " + str(bestTrain_1))
        printandwrite(pathWrite, "Best Test KNNReg Neighbor found: " + str(bestTest_1))
        f1Best, accBest, aucBest = experiment(expType, preprocess, bestTrain_1, bestTest_1, -1, -1, normalTrain,
                                              normalTest,
                                              anomalyTest, bestNeighbor, normalTrainInst, normalTestInst, anomalyInst,
                                              pathWrite, pathMatrix, pathAUC, pathDist, numFold=numFold, Info=True,
                                              randomseed=randSeed, lengthTest =lengthTest)

    # Autoencoder
    elif preprocess == 4:
        printandwrite(pathWrite, "Best Neighbor found: " + str(bestNeighbor))
        #printandwrite(pathWrite, "Best Train Model: " + str(bestTrain_1))
        #printandwrite(pathWrite, "Best Test Model: " + str(bestTest_1))
        f1Best, accBest, aucBest = experiment(expType, preprocess, ppParameter_1_Range, ppParameter_2_Range, -1, -1, normalTrain,
                                              normalTest,
                                              anomalyTest, bestNeighbor, normalTrainInst, normalTestInst, anomalyInst,
                                              pathWrite, pathMatrix, pathAUC, pathDist, numFold=numFold, Info=True,
                                              randomseed=randSeed, lengthTest =lengthTest)

    # SVD
    elif preprocess == 5:
        printandwrite(pathWrite, "Best Neighbor found: " + str(bestNeighbor))
        printandwrite(pathWrite, "Best Train Cutting Point found: " + str(bestTrain_1))
        printandwrite(pathWrite, "Best Test Cutting Point found: " + str(bestTest_1))
        f1Best, accBest, aucBest = experiment(expType, preprocess, bestTrain_1, bestTest_1, -1, -1, normalTrain,
                                              normalTest,
                                              anomalyTest, bestNeighbor, normalTrainInst, normalTestInst, anomalyInst,
                                              pathWrite, pathMatrix, pathAUC, pathDist, numFold=numFold, Info=True,
                                              randomseed=randSeed, lengthTest =lengthTest)

    # Vhi
    elif preprocess == 6:
        printandwrite(pathWrite, "Best Neighbor found: " + str(bestNeighbor))
        printandwrite(pathWrite, "Best Train KNNReg Neighbor found: " + str(bestTrain_1))
        printandwrite(pathWrite, "Best Test KNNReg Neighbor found: " + str(bestTest_1))
        f1Best, accBest, aucBest = experiment(expType, preprocess, bestTrain_1, bestTest_1, -1, -1, normalTrain,
                                              normalTest,
                                              anomalyTest, bestNeighbor, normalTrainInst, normalTestInst, anomalyInst,
                                              pathWrite, pathMatrix, pathAUC, pathDist, numFold=numFold, Info=True,
                                              randomseed=randSeed, lengthTest =lengthTest)

    # SVD-Transfer
    elif preprocess == 7:
        printandwrite(pathWrite, "Best Neighbor found: " + str(bestNeighbor))
        printandwrite(pathWrite, "Best Train KNNReg Neighbor found: " + str(bestTrain_1))
        printandwrite(pathWrite, "Best Test KNNReg Neighbor found: " + str(bestTest_1))
        f1Best, accBest, aucBest = experiment(expType, preprocess, bestTrain_1, bestTest_1, -1, -1, normalTrain,
                                              normalTest,
                                              anomalyTest, bestNeighbor, normalTrainInst, normalTestInst, anomalyInst,
                                              pathWrite, pathMatrix, pathAUC, pathDist, numFold=numFold, Info=True,
                                              randomseed=randSeed, lengthTest =lengthTest)

    # Vhi-Transfer
    elif preprocess == 8:
        printandwrite(pathWrite, "Best Neighbor found: " + str(bestNeighbor))
        printandwrite(pathWrite, "Best Train KNNReg Neighbor found: " + str(bestTrain_1))
        printandwrite(pathWrite, "Best Test KNNReg Neighbor found: " + str(bestTest_1))
        f1Best, accBest, aucBest = experiment(expType, preprocess, bestTrain_1, bestTest_1, -1, -1, normalTrain,
                                              normalTest,
                                              anomalyTest, bestNeighbor, normalTrainInst, normalTestInst, anomalyInst,
                                              pathWrite, pathMatrix, pathAUC, pathDist, numFold=numFold, Info=True,
                                              randomseed=randSeed, lengthTest =lengthTest)

    # Wavelet
    elif preprocess == 9:
        printandwrite(pathWrite, "Best Neighbor found: " + str(bestNeighbor))
        printandwrite(pathWrite, "Best Train Level found: " + str(bestTrain_1))
        printandwrite(pathWrite, "Best Test Level found: " + str(bestTest_1))
        printandwrite(pathWrite, "Best Train Wavelet Formula found: " + str(bestTrain_2))
        printandwrite(pathWrite, "Best Test Wavelet Formula found: " + str(bestTest_2))
        f1Best, accBest, aucBest = experiment(expType, preprocess, bestTrain_1, bestTest_1, bestTrain_2, bestTest_2,
                                              normalTrain,
                                              normalTest,
                                              anomalyTest, bestNeighbor, normalTrainInst, normalTestInst, anomalyInst,
                                              pathWrite, pathMatrix, pathAUC, pathDist, numFold=numFold, Info=True,
                                              randomseed=randSeed, lengthTest=lengthTest)

    else:
        printandwrite(pathWrite, "Best Neighbor found: " + str(bestNeighbor))
        f1Best, accBest, aucBest = experiment(expType, preprocess, ppParameter_1_Range[0], ppParameter_1_Range[0], 
                                              ppParameter_2_Range[0], ppParameter_2_Range[0],
                                              normalTrain, normalTest,
                                              anomalyTest, bestNeighbor, normalTrainInst, normalTestInst, anomalyInst,
                                              pathWrite, pathMatrix, pathAUC, pathDist, numFold=numFold, Info=True,
                                              randomseed=randSeed, lengthTest =lengthTest)

    stopSec = timeit.default_timer()
    printandwrite(pathWrite, 'Time for one experiment: ' + str(stopSec - startSec))

    stop = timeit.default_timer()
    printandwrite(pathWrite, 'Time for all Searches: ' + str(stop - start))


    return f1Best, accBest, aucBest