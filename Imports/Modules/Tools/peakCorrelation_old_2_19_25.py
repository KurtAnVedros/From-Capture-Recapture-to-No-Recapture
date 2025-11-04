#!/usr/bin/env python
# coding: utf-8

# # Jupyter 

# In[2]:


from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import random
import tensorflow as tf
import numpy as np
import timeit
import time
from sklearn.neighbors import NearestNeighbors
from sklearn import metrics
import scipy.io as sio
import csv
import os
import math 
import random as rand
from numpy import genfromtxt
from numpy import sum,isrealobj,sqrt
from numpy.random import standard_normal

from Modules.Evaluations import ConfidenceEstimation
from Modules.Graphs import ROC

from scipy.signal.signaltools import wiener
from sklearn.neighbors import KNeighborsRegressor
import warnings
from scipy.signal import savgol_filter
from scipy.spatial import distance
#from fastdtw import fastdtw

from sklearn.preprocessing import MinMaxScaler
from IO import Writer

from Modules.Tools import MapTool
from Modules.Tools import BoxPlot


# In[ ]:


def NormalizeData(data, Min, Max):
    return (data - Min) / (Max - Min)


# In[ ]:


# normalizes the whole dataset
def normalize(dataset):
    maxi = np.max(dataset)
    mini = np.min(dataset)
    N_dataset = np.copy(dataset)
    for s in range(dataset.shape[0]):
        for f in range(dataset.shape[1]):
            N_dataset[s][f] = (dataset[s][f] - mini)/(maxi - mini)
    return N_dataset


# In[ ]:


def maximum(dataset):
    maxi = 0
    for s in range(dataset.shape[0]):
        for f in range(len(dataset[s])):
            if dataset[s][f] > maxi:
                maxi = dataset[s][f]
    
    return maxi


# In[ ]:


def minimum(dataset):
    mini = 100000000000000
    for s in range(dataset.shape[0]):
        for f in range(len(dataset[s])):
            if dataset[s][f] < mini:
                mini = dataset[s][f]
    
    return mini


# In[ ]:


# normalizes the whole dataset
def normalizeUE(dataset):
    maxi = maximum(dataset)
    mini = minimum(dataset)
    N_dataset = np.copy(dataset)
    for s in range(dataset.shape[0]):
        for f in range(len(dataset[s])):
            N_dataset[s][f] = (dataset[s][f] - mini)/(maxi - mini)
    return N_dataset


# In[ ]:


# normalizes the whole dataset without taking into account some peaks from the start and end.
def normalizeSE(dataset, fromStart, fromEnd, pad):
    count = MapTool.getPeaks(dataset[0], pad).shape[0]
    start = fromStart
    end = count-1-fromEnd
    
    tempDataset = getSeqSignals(dataset, start, end, pad)
    
    N_dataset = normalizeByOther(dataset, tempDataset)

    return N_dataset


# In[ ]:


# normalizes the whole dataset without taking into account some peaks from the start and end.
def normalizeSec(dataset, start, end, pad, convert=""):
    if convert == "":
        convertType = 0
    elif convert == "average":
        convertType = 1
    elif convert == "median":
        convertType = 2
    
    tempDataset = getSeqSignals(dataset, start, end, pad)
    
    N_dataset = normalizeByOther(dataset, tempDataset, pad, convertType)

    return N_dataset


# In[ ]:


'''
Normalizes the dataset by taking into account only a section of the dataset.
Then the max and min values are determined by
'''
def normalizeByOther(convertDataset, limitDataset, pad, convertType=0):
    
    if convertType == 0:
        maxi = maximum(limitDataset)
        mini = minimum(limitDataset)
    
    elif convertType == 1:
        Peaks = MapTool.getPeaksDataset(limitDataset, pad)
    
        avg = []

        for p in range(len(Peaks[0])):
            avg.append(np.average(Peaks[:,p]))

        avg = np.array(avg)
    
        maxi = np.max(avg)
        mini = 0
        
    elif convertType == 2:
        Peaks = MapTool.getPeaksDataset(limitDataset, pad)

        median = []

        for p in range(len(Peaks[0])):
            median.append(np.median(Peaks[:,p]))

        median = np.array(median)
    
        maxi = np.max(median)
        mini = 0
    
    print(maxi)
    print(mini)
    
    N_dataset = np.copy(convertDataset)
    for s in range(convertDataset.shape[0]):
        for f in range(len(convertDataset[s])):
            N_dataset[s][f] = (convertDataset[s][f] - mini)/(maxi - mini)
            
    return N_dataset


# In[ ]:


'''
def rollingEuclidean_Peaks(FullSignal, SectionSignals, lowestMin, pad, amountPeaks, topNumCorr):
        
    rollingEuclideanList = []
    
    tempSect = MapTool.getPeaks(SectionSignals[0], pad)

    peakLengthSec = MapTool.getPeaks(SectionSignals[0], pad).shape[0] # 1 for the start and end points
    
    peaks = MapTool.getPeaks(FullSignal, pad).shape[0] - MapTool.getPeaks(SectionSignals[0],pad).shape[0] +1
    
    for i in range(peaks):
        j = MapTool.getPeaksLoc(FullSignal, pad, i)
        k = MapTool.getPeaksLoc(FullSignal, pad, i+amountPeaks-1)
        
        part = FullSignal[j:k]
        
        part_Peaks = MapTool.getPeaks(part, pad)

        corrs = []

        for s in range(SectionSignals.shape[0]):           
            sect = SectionSignals[s]
            SectSignal_Peaks = MapTool.getPeaks(sect, pad)

            distSect = distance.euclidean(part_Peaks, SectSignal_Peaks)
            distSect = distSect * -1

            # collect the correlation score.
            corrs.append(distSect)

        # go through various top number of corr to collect the sum and adverage.
        corrs = np.array(corrs)
        topCorrsInd = np.argpartition(corrs, -topNumCorr)[-topNumCorr:]
        topCorrs = corrs[topCorrsInd]
        topCorrsInd = topCorrs.argsort()[:][::-1]
        topCorrs = topCorrs[topCorrsInd]
        
        topCorr = np.mean(topCorrs)
        
        rollingEuclideanList.append(topCorr)
        
    rollingEuclideanList = np.array(rollingEuclideanList, dtype=object)
    
    Min = lowestMin
    Max = 0
    
    rollingEuclideanBestCorrelation = NormalizeData(rollingEuclideanList, Min, Max)
    
    return rollingEuclideanBestCorrelation  
'''


# In[ ]:


'''
Method for Normalized Euclidean Distance. This is from the following formula.
normalized Squared Euclidean Distance (NSED): 0.5 * (var(X-Y))/(var(X)+ var(Y))  where var(X) = abs((X-mean(X))^2)
normalized Euclidean Distance: NSED ** 0.5

X: numpy array of values.
Y: numpy array of values.
'''
def N_EuclideanDistance(X, Y):
    nsed = 0.5*((np.var(X-Y))/(np.var(X)+np.var(Y)))
    ned = nsed ** 0.5
    
    return ned


# In[ ]:


'''
Method for Euclidean Distance. Makes use of SciPy formula

X: numpy array of values.
Y: numpy array of values.
'''
def EuclideanDistance(X, Y):
    ed = distance.euclidean(X, Y)
    
    return ed


# In[ ]:


'''
Method for cosine similarity. Makes use of SciPy formula.

X: numpy array of values.
Y: numpy array of values.
'''
def cosineSimilarity(X, Y):
    cs = distance.cosine(X, Y)
    
    return cs


# In[ ]:


'''
Method takes the base code and sequences of code to identify the indexes where 
these sequences show in the overall program code.

Base_Code: full program code given in sequences.
sequences: the sequences to identify given in a list object.
'''
def obtainRemoveIndexes(Base_Code, sequences, tokenizer):
    token_Seq = []
    # Convert sequences to tokens
    for seq in sequences:
        token_Seq.append(tokenizer.texts_to_sequences(seq)[0])
    
    # Identify indexes
    Remove_Indexes = []
    for seq in token_Seq:
        for i in range(len(Base_Code)):
            if (Base_Code[i] == seq).all():
                Remove_Indexes.append(i)
    
    return Remove_Indexes


# In[ ]:


'''
Method removes indexes from a List object.

List: list to remove from. Does not affect original given list.
RemoveIndexes: Indexes to remove given as a list object.
'''
def removeFromList(List, RemoveIndexes):
    newList = List[:]
    newList = newList.tolist()
    indicesList = sorted(RemoveIndexes, reverse=True)
    
    
    # Traversing in the indices list
    for indx in indicesList:
       # checking whether the corresponding iterator index is less than the list length
       if indx < len(newList):
          # removing element by index using pop() function
          newList.pop(indx)
            
    return np.array(newList)


# In[ ]:


'''
This method is used to perform Euclidean Distance on certain peak. 

TS_Signal: (Time Series Signal) the EM signal of an execution path.
Q_Signals: (Query Signal) the EM signals of a set of instructions.
Pad: Used for collecting the peaks. Is the average number of features between high and low peaks.
peakLoc: the location of the peak to do Euclidean Distance on.
'''
def ED_Peak(TS_Signal, Q_Signals, MM_Scaler, Pad, peak, toNormalize):
    # Create a list of the Euclidean Distances.
    ED_List = []
    
    # Obtain the peak of the TS_Signal to do similarity.
    peakLoc = MapTool.getPeaksLoc(TS_Signal, Pad, peak)
    
    TS_Peak = TS_Signal[peakLoc]
    
    # Go through each Query Signal
    for s in range(Q_Signals.shape[0]):    
        # Select a Query Signal and get only the peak.
        Q_Signal = Q_Signals[s]
        Q_Signal_PeakLoc = MapTool.getPeaksLoc(Q_Signal, Pad, peak)
        Q_Peak = Q_Signal[Q_Signal_PeakLoc]
        
        if toNormalize == 1:
            # perform Min Max scaling on part Time Series Signal and Query Signal
            MM_TS_Peaks = MM_Scaler.transform(np.reshape(TS_Signal, (-1,1)))
            MM_Q_Signal_Peaks = MM_Scaler.transform(np.reshape(Q_Signal, (-1,1)))
            
            TS_Peak = MM_TS_Peaks[peakLoc]
            Q_Peak = MM_Q_Signal_Peaks[Q_Signal_PeakLoc]
            
        # Perform Euclidean Distance on the peaks.
        dist = EuclideanDistance(np.array([TS_Peak]), np.array([Q_Peak]))
        
        # collect the Euclidean Distance score.
        ED_List.append(dist)
    
    return ED_List


# In[ ]:


'''
This method is used to perform Euclidean Distance on certain peak. 

Assumes the peaks are given

TS_Peak: (Time Series Signal) the EM signal of an execution path.
Q_Peaks: (Query Signal) the EM signals of a set of instructions.
Pad: Used for collecting the peaks. Is the average number of features between high and low peaks.
peakLoc: the location of the peak to do Euclidean Distance on.
'''
def ED_Peak_OnlyPeaks(TS_Peak, Q_Peaks, MM_Scaler, Pad, peak, toNormalize):
    # Create a list of the Euclidean Distances.
    ED_List = []
    
    # Obtain the peak of the TS_Peak to do similarity.    
    TS_Peak_value = TS_Peak[peak]
    
    # Go through each Query Signal
    for s in range(Q_Peaks.shape[0]):    
        # Select a Query Signal and get only the peak.
        Q_Peak = Q_Peaks[s]
        Q_Peak_value = Q_Peak[peak]
            
        # Perform Euclidean Distance on the peaks.
        dist = EuclideanDistance(np.array([TS_Peak_value]), np.array([Q_Peak_value]))
        
        # collect the Euclidean Distance score.
        ED_List.append(dist)
    
    return ED_List


# In[ ]:


'''
This method is used to perform sliding window similarities. 
The similarities available is 
0) flipped normalized Euclidean Distance: uses euclidean distance then negatives the values. Then Min Max.
1) Normalized Euclidean Distance.
2) Cosine SImilarity
3) Min-Max scale preprocessing Euclidean Distance.
4) Basic Euclidean Distance
5) Cross Correlation on peaks
6) Cross Correlation on full signals (takes about 10x longer than on peaks only).

TS_Signal: (Time Series Signal) the EM signal of an execution path.
Q_Signals: (Query Signal) the EM signals of a set of instructions.
MM_Scaler: a MinMax scaler already fitted to the peaks of the time series siganls and Query Signals
Pad: Used for collecting the peaks. Is the average number of features between high and low peaks.
amountPeaks: The number of peaks that the Query Signal has. 
typeSimM: The type of similarity measurment done
    0) flipped normalized Euclidean Distance: uses euclidean distance then negatives the values. Then Min Max.
    1) Normalized Euclidean Distance.
    2) Cosine SImilarity
    3) Min-Max scale preprocessing Euclidean Distance.
Remove_Peak_List: Peaks to remove before comparing.
topNumSim: The number of similarities averaged together. Goes by the top amounts. 
'''
def slidingWIndow_Peaks(
    TS_Signal, QE_Signals, MM_Scaler, Pad, amountPeaks, typeSimM, topNumSim,
    Remove_Peak_List=[], nonPeakEnds=False, PeakOnlyData=False
):
    import numpy as np
    from scipy.spatial import distance

    # Determine the number of peaks to iterate over
    if typeSimM != 6:
        if PeakOnlyData:
            peaks = TS_Signal.shape[0] - QE_Signals[0].shape[0] + 1
        else:
            TS_Peaks = MapTool.getPeaks(TS_Signal, Pad)
            QE_Peaks = MapTool.getPeaks(QE_Signals[0], Pad)
            peaks = TS_Peaks.shape[0] - QE_Peaks.shape[0] + 1

        # Precompute peaks for all query signals if needed
        if not PeakOnlyData:
            QE_Signal_Peaks = [MapTool.getPeaks(q_signal, Pad) for q_signal in QE_Signals]
        else:
            QE_Signal_Peaks = QE_Signals

        if Remove_Peak_List:
            QE_Signal_Peaks = [removeFromList(peaks, Remove_Peak_List) for peaks in QE_Signal_Peaks]

        slidingWindowList = []

        # Iterate over the peaks of the time-series signal
        for i in range(peaks):
            if nonPeakEnds:
                k = i + amountPeaks - 2
                part = TS_Signal[:k] if PeakOnlyData else TS_Signal[:MapTool.getPeaksLoc(TS_Signal, Pad, k) + 1]
            else:
                j = i if PeakOnlyData else MapTool.getPeaksLoc(TS_Signal, Pad, i)
                k = i + amountPeaks if PeakOnlyData else MapTool.getPeaksLoc(TS_Signal, Pad, i + amountPeaks) + 1
                part = TS_Signal[j:k]

            # Extract peaks
            part_Peaks = part if PeakOnlyData else MapTool.getPeaks(part, Pad)
            if Remove_Peak_List:
                part_Peaks = removeFromList(part_Peaks, Remove_Peak_List)

            # Compute similarities with all query signals
            sims = []
            for q_peaks in QE_Signal_Peaks:
                if typeSimM == 0:  # Flipped normalized Euclidean Distance
                    dist = -distance.euclidean(part_Peaks, q_peaks)
                elif typeSimM == 1:  # Normalized Euclidean Distance
                    dist = N_EuclideanDistance(part_Peaks, q_peaks)
                elif typeSimM == 2:  # Cosine Similarity
                    dist = cosineSimilarity(part_Peaks, q_peaks)
                elif typeSimM == 3:  # Min-Max scale preprocessing Euclidean Distance
                    MM_part_Peaks = MM_Scaler.transform(part_Peaks.reshape(-1, 1)).flatten()
                    MM_Q_Peaks = MM_Scaler.transform(q_peaks.reshape(-1, 1)).flatten()
                    dist = distance.euclidean(MM_part_Peaks, MM_Q_Peaks)
                elif typeSimM == 4:  # Basic Euclidean Distance
                    dist = distance.euclidean(part_Peaks, q_peaks)
                elif typeSimM == 5 or typeSimM == 6:  # Cross Correlation
                    dist = np.max(np.correlate(part_Peaks, q_peaks, mode='valid'))
                sims.append(dist)

            # Take the average of the top similarities
            sims = np.array(sims)
            top_indices = np.argpartition(sims, topNumSim)[:topNumSim]
            top_sims = sims[top_indices]
            topSim = np.mean(top_sims)

            slidingWindowList.append(topSim)
    
    else:
        slidingWindowList = []

        # Iterate over the peaks of the time-series signal
        sims = []
        for QE_Signal in QE_Signals:
            # Compute similarities with all query signals
            if typeSimM == 5 or typeSimM == 6:  # Cross Correlation
                dist = np.max(np.correlate(TS_Signal, QE_Signal, mode='valid'))
            sims.append(dist)

        # Take the average of the top similarities
        sims = np.array(sims)
        top_indices = np.argpartition(sims, topNumSim)[len(sims)-topNumSim:len(sims)]
        top_sims = sims[top_indices]
        topSim = np.mean(top_sims)

        slidingWindowList.append(topSim)

    slidingWindowList = np.array(slidingWindowList, dtype=object)

    # Normalize for flipped normalized Euclidean Distance
    if typeSimM == 0:
        slidingWindowList = NormalizeData(slidingWindowList, -200, 0)

    return slidingWindowList


'''
old Method, retired due to slow time and redundancy

def slidingWIndow_Peaks(TS_Signal, QE_Signals, MM_Scaler, Pad, amountPeaks, typeSimM, topNumSim, 
                               Remove_Peak_List = [], nonPeakEnds = False, PeakOnlyData = False):
    
    # Create a list of the similarities.
    slidingWindowList = []
    
    # number of peaks to check though on the Time Series Signal. Note that it from start to the end - length of the Query Signal.
    if PeakOnlyData:
        peaks = TS_Signal.shape[0] - QE_Signals[0].shape[0]+1
    else:
        peaks = MapTool.getPeaks(TS_Signal, Pad).shape[0] - MapTool.getPeaks(QE_Signals[0],Pad).shape[0] +1
    
    # Go through each peak of the Time Series Signal.
    for i in range(peaks):       
        if nonPeakEnds:
            # Obtain the section of the TS_Signal to do similarity.
            if PeakOnlyData:
                k = i+amountPeaks-2
            else:
                k = MapTool.getPeaksLoc(TS_Signal, Pad, i+amountPeaks-2)+1
        
            part = TS_Signal[:k]
        
            Q_Signals = []
            for signal in QE_Signals:
                if PeakOnlyData:
                    k = i+amountPeaks-2
                else:
                    k = MapTool.getPeaksLoc(signal, Pad, i+amountPeaks-2)+1
                
                Q_Signals.append(signal[:k])
            
            Q_Signals = np.array(Q_Signals, dtype=object)
            
        else:
            # Obtain the section of the TS_Signal to do similarity.
            if PeakOnlyData:
                j = i
                k = i+amountPeaks
            else:
                j = MapTool.getPeaksLoc(TS_Signal, Pad, i)
                k = MapTool.getPeaksLoc(TS_Signal, Pad, i+amountPeaks)+1

            part = TS_Signal[j:k]
        
            Q_Signals = np.copy(QE_Signals)
        
        # Convert to peaks only for the Time Series signal
        if PeakOnlyData:
            part_Peaks = part
        else:
            part_Peaks = MapTool.getPeaks(part, Pad)
            
        # Remove peaks before comparing.
        if len(Remove_Peak_List) > 0:
            part_Peaks = removeFromList(part_Peaks, Remove_Peak_List)
    
        # Create a list for the similarity between Time Series signal and all the Query Signals.
        sims = []

        # Go through each Query Signal
        for s in range(Q_Signals.shape[0]):    
            # Select a Query Signal and get only the peaks.
            Q_Signal = Q_Signals[s]
            if PeakOnlyData:
                Q_Signal_Peaks = Q_Signal
            else:
                Q_Signal_Peaks = MapTool.getPeaks(Q_Signal, Pad)
                
            # Remove peaks before comparing.
            if len(Remove_Peak_List) > 0:
                Q_Signal_Peaks = removeFromList(Q_Signal_Peaks, Remove_Peak_List)

            # Perform one of the sliding window similarity options.
            ## 0) performs flipped normalized Euclidean Distance
            if typeSimM == 0:
                dist = distance.euclidean(part_Peaks, Q_Signal_Peaks)
                dist = dist * -1
            
            ## 1) performs Normalized Euclidean Distance
            elif typeSimM == 1:
                dist = N_EuclideanDistance(part_Peaks, Q_Signal_Peaks)
            
            ## 2) performs Cosine Similarity
            elif typeSimM == 2:
                dist = cosineSimilarity(part_Peaks, Q_Signal_Peaks)
            
            ## 3) performs Min-Max scale preprocessing Euclidean Distance
            elif typeSimM == 3:
                # perform Min Max scaling on part Time Series Signal and Query Signal
                MM_part_Peaks = MM_Scaler.transform(np.reshape(part_Peaks, (-1,1)))
                MM_Q_Signal_Peaks = MM_Scaler.transform(np.reshape(Q_Signal_Peaks, (-1,1)))
                
                MM_part_Peaks = MM_part_Peaks.flatten()
                MM_Q_Signal_Peaks = MM_Q_Signal_Peaks.flatten()
                
                dist = EuclideanDistance(MM_part_Peaks, MM_Q_Signal_Peaks)
            
            ## 4) basic Euclidean Distance
            elif typeSimM == 4:
                dist = EuclideanDistance(part_Peaks, Q_Signal_Peaks)
                

            # collect the similarity score.
            sims.append(dist)

        # go through various top number of similarities and obtain the mean.
        ## this is to reduce the effect of outlier query signals.
        ## This is for getting the top distances.
        #sims = np.array(sims)
        #print(sims)
        #topSimsInd = np.argpartition(sims, -topNumSim)[-topNumSim:]
        #topSims = sims[topSimsInd]
        #topSimsInd = topSims.argsort()[:][::-1]
        #topSims = topSims[topSimsInd]
        #print(topSims)
        
        #topSim = np.mean(topSims)
        ## this is for getting the lowest distances.
        sims = np.array(sims)
        topSimsInd = np.argpartition(sims, topNumSim)[:topNumSim]
        topSims = sims[topSimsInd]
        topSimsInd = topSims.argsort()[:][::1]
        topSims = topSims[topSimsInd]
        
        topSim = np.mean(topSims)
        
        # place inside a list of the similarities
        slidingWindowList.append(topSim)
        
    slidingWindowList = np.array(slidingWindowList, dtype=object)
    
    # last part of flipped normalized Euclidean Distance.
    ## this normalizes the similarities to a set standard for Min and Max euclidean distance.
    if typeSimM == 0:
        Min = -200
        Max = 0
    
        slidingWindowList = NormalizeData(slidingWindowList, Min, Max)
    
    return slidingWindowList
'''

# In[ ]:


def plotSimilarity(similarity, titleName, toFileSave, location):
    ### Plot the correlation
    plt.plot(similarity)
    plt.title(titleName)
    plt.xlabel("Starting Peak in Time Sereies Signal")
    plt.ylabel("Similarity Measurement")
    if toFileSave == 0:
        plt.show()
    elif toFileSave == 1:
        try:
            plt.savefig(location)
        except FileNotFoundError as error:
            logging.error(error)
    elif toFileSave == 2:
        try:
            plt.savefig(location)
        except FileNotFoundError as error:
            logging.error(error)
        plt.show()


# In[ ]:


def plotMostLikelyArea(TS_Signal, Q_Signals, sims, TS_Signal_name, Q_Signal_name, titleName, pad, amountPeaks, 
                       toFileSave, location):
    ### Plot the most likely area in the signal sample

    peakNum = np.argmin(sims)
    start = MapTool.getPeaksLoc(TS_Signal, pad, peakNum)
    sim = min(sims)
    end = MapTool.getPeaksLoc(TS_Signal, pad, peakNum+amountPeaks-1)
    QInSample = TS_Signal[start:end]

    time_Signal = np.arange(TS_Signal.shape[0])
    time_Seq = np.arange(start,end)

    plt.plot(time_Signal, TS_Signal, label = "Time-Series " + TS_Signal_name)
    plt.plot(time_Seq, QInSample, label = "Query " +  Q_Signal_name)
    plt.title(titleName)
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.legend()
    if toFileSave == 0:
        plt.show()
    elif toFileSave == 1:
        try:
            plt.savefig(location)
        except FileNotFoundError as error:
            logging.error(error)
    elif toFileSave == 2:
        try:
            plt.savefig(location)
        except FileNotFoundError as error:
            logging.error(error)
        plt.show()


# In[ ]:


def getSeqSignals(Signals, peaksStart, peaksEnd, pad, startHigh=True):
    seqList = []

    for signal in Signals:
        if startHigh==True:
            Seq = signal[MapTool.getPeaksLoc(signal, pad, peaksStart):MapTool.getPeaksLoc(signal, pad, peaksEnd)]
        else:
            Seq = signal[MapTool.getPeaksLoc(signal, pad, peaksStart, getHigh = False):MapTool.getPeaksLoc(signal, pad, peaksEnd)]
        seqList.append(Seq)
    
    seqList = np.array(seqList, dtype="object")
    return seqList


# In[ ]:


'''
def getEvaluations(corrs, ground_truths, numberThresholds=1000):
    results = []

    for threshold in np.arange(0, 1 + 1 / numberThresholds, 1 / numberThresholds):
        predictions = []
        for corr in corrs:
            for index in range(corr.shape[0]):
                if corr[index] > threshold:
                    predictions.append(1)
                else:
                    predictions.append(0)
        
        predictions = np.array(predictions)
        
        ground_truth = ground_truths.flatten()

        metrics = ConfidenceEstimation.calculateBasicMetrics(ground_truth, predictions)

        results.append(metrics)

    return results
'''


# In[ ]:


def getEvaluations(sims, ground_truths, numberThresholds=1000):
    results = []

    for threshold in np.arange(0, 1 + 1 / numberThresholds, 1 / numberThresholds):
        predictions = []
        for sim in sims:
            for index in range(sim.shape[0]):
                if sim[index] > threshold:
                    predictions.append(0)
                else:
                    predictions.append(1)
        
        predictions = np.array(predictions)
        
        ground_truth = ground_truths.flatten()

        metrics = ConfidenceEstimation.calculateBasicMetrics(ground_truth, predictions)

        results.append(metrics)

    return results


# In[ ]:


def getEvaluationsOne(corr, ground_truth, numberThresholds=1000):
    results = []

    for threshold in np.arange(0, 1 + 1 / numberThresholds, 1 / numberThresholds):
        predictions = []
        for index in range(corr.shape[0]):
            if corr[index] > threshold:
                predictions.append(1)
            else:
                predictions.append(0)
        
        predictions = np.array(predictions)

        metrics = ConfidenceEstimation.calculateBasicMetrics(ground_truth, predictions)

        results.append(metrics)

    return results


# In[ ]:


# Methods for writing to a file. Two exists. One that will only write to a file and one that will write to file and output
# to jupyter Notebook.

def printOrWriteInfo(location, text, toFileSave, firstLine = False):
    if toFileSave ==0:
        print(text)
    elif toFileSave == 1:
        if firstLine:
            sourceFile = open(location, 'w')
        else:
            sourceFile = open(location, 'a')
        sourceFile.write(text)
        sourceFile.close()
    elif toFileSave == 2:
        if firstLine:
            sourceFile = open(location, 'w')
        else:
            sourceFile = open(location, 'a')
        sourceFile.write(text)
        print(text)
        sourceFile.close()


# In[ ]:


'''
Performs the experiment of comparing the instructions main peaks to determin the number of peaks influenced by prior instruction/s.

TS_Signals_G: (Time Series Signals) the EM signals of an execution path.
Q_Signals_G: (Query Signals) the EM signals of an set instruction/s.
TS_Signal_name: the name of the time series signal to be displayed in graphs
Q_Signal_name: the name of the query signal to be displayed in graphs
Pad: Used for collecting the peaks. Is the average number of features between high and low peaks.
dictionary: contains information on the instructions. (peakLoc, name)
'''

def instComparisonExp(TS_Signals_G, Q_Signals_G, TS_Signal_name, Q_Signal_name, Pad, dictionary,
                     y_limit_low=0, y_limit_high=0, toNormalize = 0):
    
    TS_Signals = TS_Signals_G
    Q_Signals = Q_Signals_G
    
    # Create the Min-Max Scalar
    MM_Scaler = MinMaxScaler()
    
    # normalize the two datasets
    if toNormalize == 1:       
        # gather all the peaks of the Time Series Signals and Query Signals
        TS_Signals_Peaks =  MapTool.getPeaksDataset(TS_Signals, Pad)
        Q_Signals_Peaks =  MapTool.getPeaksDataset(Q_Signals, Pad)
        
        # Find the max and min of the two.
        TS_Max = np.max(TS_Signals_Peaks)
        Q_Max = np.max(Q_Signals_Peaks)
        TS_Min = np.min(TS_Signals_Peaks)
        Q_Min = np.mmin(Q_Signals_Peaks)
        
        # place into array
        info = np.array([TS_Max, Q_Max, TS_Min, Q_Min])
        info = np.reshape(info, (-1,1)) 
        
        # Create the Min-Max Scalar based on the min max of the info collected.
        MM_Scaler.fit(info)
    
    print("Beginning Tests...")
    ED_List_All =[]
    
    # Gather the Euclidean Distance of each instruction comparison
    ## go through each instruction
    for i in range(dictionary.shape[0]):
        peaks = dictionary[i][0]
        name =  dictionary[i][1]
        
        print("Working on Instruction: "+ name)
        start = timeit.default_timer()
        
        ED_Inst_List = []
        
        # Go through each Time-Series Sample
        for t in range(TS_Signals.shape[0]):

            TS_Signal = TS_Signals[t]

            # Obtain the Euclidean DIstances across the Time-Series Sample given the Query signals for the peaks
            ED_TS_Sample_List = ED_Peak(TS_Signal, Q_Signals, MM_Scaler, Pad, peaks, toNormalize)
            ED_TS_Sample_List = np.array(ED_TS_Sample_List)
            
            for p in range(ED_TS_Sample_List.shape[0]):
                ED_Inst_List.append(ED_TS_Sample_List[p])
        
        ED_Inst_List = np.array(ED_Inst_List)
        print(ED_Inst_List.shape)
        
        rowInfo = np.array([peaks, name, ED_Inst_List], dtype=object)
        ED_List_All.append(rowInfo)
        
        stop = timeit.default_timer()

        print('Time: ', stop - start)  
            
    ED_List_All = np.array(ED_List_All)
    
    # Plot the max Euclidean Distance distance of all the instructions
    plotDistances = []
    plotXNames = []
    
    for n in range(ED_List_All.shape[0]):
        plotDistances.append(np.max(ED_List_All[n][2]))
        plotXNames.append(ED_List_All[n][1])
    
    plotDistances = np.array(plotDistances)
    plotXNames = np.array(plotXNames)
    
    titleName = "Maximum Distances between the TS and Q Samples"
    
    plt.plot(plotDistances)
    plt.title(titleName)
    
    plt.xlabel("Instructions")
    plt.ylabel("Euclidean Distance")
    
    x = np.arange(plotDistances.shape[0])
    
    plt.xticks(x, plotXNames)
    # Pad margins so that markers don't get clipped by the axes
    plt.margins(0.2)
    # Tweak spacing to prevent clipping of tick-labels
    plt.subplots_adjust(bottom=0.15)
    plt.show()
    
    
    # Plot the average Euclidean Distance of all the instructions.
    plotDistances = []
    
    for n in range(ED_List_All.shape[0]):
        plotDistances.append(np.mean(ED_List_All[n][2]))
    
    plotDistances = np.array(plotDistances)
    
    titleName = "Average Distances between the TS and Q Samples"
    
    plt.plot(plotDistances)
    plt.title(titleName)
    
    plt.xlabel("Instructions")
    plt.ylabel("Euclidean Distance")
    
    x = np.arange(plotDistances.shape[0])
    
    plt.xticks(x, plotXNames)
    # Pad margins so that markers don't get clipped by the axes
    plt.margins(0.2)
    # Tweak spacing to prevent clipping of tick-labels
    plt.subplots_adjust(bottom=0.15)
    plt.show()
    
    # Plot the min Euclidean Distance distance of all the instructions.
    plotDistances = []
    
    for n in range(ED_List_All.shape[0]):
        plotDistances.append(np.min(ED_List_All[n][2]))
    
    plotDistances = np.array(plotDistances)
    
    titleName = "Minimum Distances between the TS and Q Samples"
    
    plt.plot(plotDistances)
    plt.title(titleName)
    
    plt.xlabel("Instructions")
    plt.ylabel("Euclidean Distance")
    
    x = np.arange(plotDistances.shape[0])
    
    plt.xticks(x, plotXNames)
    # Pad margins so that markers don't get clipped by the axes
    plt.margins(0.2)
    # Tweak spacing to prevent clipping of tick-labels
    plt.subplots_adjust(bottom=0.15)
    plt.show()
    
    # Plot the BoxGraph of the Euclidean Distance of all instructions.
    titleName = "BoxPlot of the EDs between the TS and Q Samples"
    BoxPlot.boxPlotInstPeaks(ED_List_All, "Instructions", "Euclidean Distance", titleName, y_limit_low, y_limit_high)
    


# In[ ]:


'''
Performs the experiment of comparing the instructions main peaks to determin the number of peaks influenced by prior instruction/s.

This Method assumes the Peaks have alread been taken

TS_Peaks_G: (Time Series Signals) the EM signals of an execution path.
Q_Peaks_G: (Query Signals) the EM signals of an set instruction/s.
TS_Peak_name: the name of the time series signal to be displayed in graphs
Q_Peak_name: the name of the query signal to be displayed in graphs
Pad: Used for collecting the peaks. Is the average number of features between high and low peaks.
dictionary: contains information on the instructions. (peakLoc, name)
'''

def instComparisonExp_OnlyPeaks(TS_Peaks_G, Q_Peaks_G, TS_Peak_name, Q_Peak_name, Pad, dictionary,
                     y_limit_low=0, y_limit_high=0, toNormalize = 0):
    
    TS_Peaks = TS_Peaks_G
    Q_Peaks = Q_Peaks_G
    
    # Create the Min-Max Scalar
    MM_Scaler = MinMaxScaler()
    
    
    print("Beginning Tests...")
    ED_List_All =[]
    
    # Gather the Euclidean Distance of each instruction comparison
    ## go through each instruction
    for i in range(dictionary.shape[0]):
        peaks = dictionary[i][0]
        name =  dictionary[i][1]
        
        print("Working on Instruction: "+ name)
        start = timeit.default_timer()
        
        ED_Inst_List = []
        
        # Go through each Time-Series Sample
        for t in range(TS_Peaks.shape[0]):

            TS_Peak = TS_Peaks[t]

            # Obtain the Euclidean DIstances across the Time-Series Sample given the Query signals for the peaks
            ED_TS_Sample_List = ED_Peak_OnlyPeaks(TS_Peak, Q_Peaks, MM_Scaler, Pad, peaks, toNormalize)
            ED_TS_Sample_List = np.array(ED_TS_Sample_List)
            
            for p in range(ED_TS_Sample_List.shape[0]):
                ED_Inst_List.append(ED_TS_Sample_List[p])
        
        ED_Inst_List = np.array(ED_Inst_List)
        print(ED_Inst_List.shape)
        
        rowInfo = np.array([peaks, name, ED_Inst_List], dtype=object)
        ED_List_All.append(rowInfo)
        
        stop = timeit.default_timer()

        print('Time: ', stop - start)  
            
    ED_List_All = np.array(ED_List_All)
    
    # Plot the max Euclidean Distance distance of all the instructions
    plotDistances = []
    plotXNames = []
    
    for n in range(ED_List_All.shape[0]):
        plotDistances.append(np.max(ED_List_All[n][2]))
        plotXNames.append(ED_List_All[n][1])
    
    plotDistances = np.array(plotDistances)
    plotXNames = np.array(plotXNames)
    
    titleName = "Maximum Distances between the TS and Q Samples"
    
    plt.plot(plotDistances)
    plt.title(titleName)
    
    plt.xlabel("Instructions")
    plt.ylabel("Euclidean Distance")
    
    x = np.arange(plotDistances.shape[0])
    
    plt.xticks(x, plotXNames)
    # Pad margins so that markers don't get clipped by the axes
    plt.margins(0.2)
    # Tweak spacing to prevent clipping of tick-labels
    plt.subplots_adjust(bottom=0.15)
    plt.show()
    
    
    # Plot the average Euclidean Distance of all the instructions.
    plotDistances = []
    
    for n in range(ED_List_All.shape[0]):
        plotDistances.append(np.mean(ED_List_All[n][2]))
    
    plotDistances = np.array(plotDistances)
    
    titleName = "Average Distances between the TS and Q Samples"
    
    plt.plot(plotDistances)
    plt.title(titleName)
    
    plt.xlabel("Instructions")
    plt.ylabel("Euclidean Distance")
    
    x = np.arange(plotDistances.shape[0])
    
    plt.xticks(x, plotXNames)
    # Pad margins so that markers don't get clipped by the axes
    plt.margins(0.2)
    # Tweak spacing to prevent clipping of tick-labels
    plt.subplots_adjust(bottom=0.15)
    plt.show()
    
    # Plot the min Euclidean Distance distance of all the instructions.
    plotDistances = []
    
    for n in range(ED_List_All.shape[0]):
        plotDistances.append(np.min(ED_List_All[n][2]))
    
    plotDistances = np.array(plotDistances)
    
    titleName = "Minimum Distances between the TS and Q Samples"
    
    plt.plot(plotDistances)
    plt.title(titleName)
    
    plt.xlabel("Instructions")
    plt.ylabel("Euclidean Distance")
    
    x = np.arange(plotDistances.shape[0])
    
    plt.xticks(x, plotXNames)
    # Pad margins so that markers don't get clipped by the axes
    plt.margins(0.2)
    # Tweak spacing to prevent clipping of tick-labels
    plt.subplots_adjust(bottom=0.15)
    plt.show()
    
    # Plot the BoxGraph of the Euclidean Distance of all instructions.
    titleName = "BoxPlot of the EDs between the TS and Q Samples"
    BoxPlot.boxPlotInstPeaks(ED_List_All, "Instructions", "Euclidean Distance", titleName, y_limit_low, y_limit_high)


# In[ ]:


'''
Performs experiments with sliding window similarities between a given Time Series Signals and Query Signals.

TS_Signals: (Time Series Signals) the EM signals of an execution path.
Q_Signals: (Query Signals) the EM signals of an set instruction/s.
TS_Signal_name: the name of the time series signal to be displayed in graphs
Q_Signal_name: the name of the query signal to be displayed in graphs
Pad: Used for collecting the peaks. Is the average number of features between high and low peaks.
typeSimM: The type of similarity measurment done.
    0) flipped normalized Euclidean Distance: uses euclidean distance then negatives the values. Then Min Max.
    1) Normalized Euclidean Distance.
    2) Cosine SImilarity
    3) Min-Max scale preprocessing Euclidean Distance.
peaksToLookAt: the peaks that contains the Query signal in the given Time Series Signals.
    So far only works with ONE.
amtGraphs: the amount of sliding Window similarity Graphs to show.
topNumSim: the number of sliding Window similarities to mean for one similarity. To reduce the effects of outlier Query signals.
    Note given as array to test multiple values
'''
def slidingWindowExp(TS_Signals, Q_Signals, TS_Signal_name, Q_Signal_name, Pad, peaksToLookAt, typeSimM,
                     amtGraphs = 20, topNumSim=np.array([10]), toFileSave = 0, GeneralPath= ""):
    
    if toFileSave < 0 or toFileSave > 2:
        print("Error: non valid toFileSave value.")
        
    if toFileSave == 1 or toFileSave == 2:
        if GeneralPath == "":
            print("Error: no path specified.")
    
    ResultsFile = GeneralPath + "/results.txt"
    
    ## Create path if does not exist.
    # Check whether the specified path exists or not
    isExist = os.path.exists(GeneralPath)

    if not isExist:
      # Create a new directory because it does not exist 
      os.makedirs(GeneralPath)
    
    # Get the number of peaks in the Query Signals
    amountPeaks = MapTool.getPeaks(Q_Signals[0], Pad).shape[0]
    
    # Create the Min-Max Scalar
    MM_Scaler = MinMaxScaler()
    
    # String name of the similarity measurement.
    SM_name = ""
    if typeSimM == 0:
        SM_name = "flipped normalized Euclidean Distance"
        SM_nameShort = "FNED"
    elif typeSimM == 1:
        SM_name = "Normalized Euclidean Distance"
        SM_nameShort = "NED"
    elif typeSimM == 2:
        SM_name = "Cosine Similarity"
        SM_nameShort = "CS"
    elif typeSimM == 3:
        SM_name = "MinMax Euclidean Distance"
        SM_nameShort = "MED"
        
    ResultsFile = GeneralPath + SM_nameShort + "_TS_" + TS_Signal_name + "_Q_" + Q_Signal_name + " results.txt"
    
    printOrWriteInfo(ResultsFile, "Beginning Tests...", toFileSave, True)
    
    # check for errors:
    checkCount = []
    for Q_Signal in Q_Signals:
        checkCount.append(MapTool.getPeaks(Q_Signal, Pad).shape[0])
    
    checkCount = np.array(checkCount)
    checkCount = np.unique(checkCount)
    
    if checkCount.shape[0] > 1:
        printOrWriteInfo(ResultsFile, "Error in gathering Q_Signals as unique peaks gave different values", toFileSave)
        printOrWriteInfo(ResultsFile, checkCount, toFileSave)
    
    
    # Create a Min Max Scaler for later use with Min-Max scale preprocessing Euclidean Distance
    if typeSimM == 3:
        # gather all the peaks of the Time Series Signals and Query Signals
        TS_Signals_Peaks =  MapTool.getPeaksDataset(TS_Signals, Pad)
        Q_Signals_Peaks =  MapTool.getPeaksDataset(Q_Signals, Pad)
        
        # Find the max and min of the two.
        TS_Max = np.max(TS_Signals_Peaks)
        Q_Max = np.max(Q_Signals_Peaks)
        TS_Min = np.min(TS_Signals_Peaks)
        Q_Min = np.min(Q_Signals_Peaks)
        
        # place into array
        info = np.array([TS_Max, Q_Max, TS_Min, Q_Min])
        info = np.reshape(info, (-1,1)) 
        
        # Create the Min-Max Scalar based on the min max of the info collected.
        MM_Scaler.fit(info)
    
    # Set the number of the top number of sliding Window similarities.
    ## this will go through each to test each number given
    for number in topNumSim:
        start = timeit.default_timer()
    
        count = 0
        correct =  0
        SW_Sims = []
        
        
        printOrWriteInfo(ResultsFile, " ", toFileSave)
        printOrWriteInfo(ResultsFile, "Taking the top number of sliding Window similarities: " + str(number), toFileSave)
        
        # Get a Time Series Signal for similarity measurement
        for i in range(TS_Signals.shape[0]):

            TS_Signal = TS_Signals[i]

            # Find out if to graph the sliding window similarity measurement for the time series signal
            ## This case graph
            if i < amtGraphs:
                # Title of section
                if toFileSave == 0 or toFileSave == 2:
                    print(TS_Signal_name + ": Signal Sample " + str(i))

                # Obtain the sliding window similarity across the time series signal given the query signals
                sims = slidingWIndow_Peaks(TS_Signal, Q_Signals, MM_Scaler, Pad, amountPeaks, typeSimM, number)
                SW_Sims.append(sims)

                ### Plot the correlation
                simFile = "Sim_" + SM_nameShort + "_TS_" + TS_Signal_name + "_Sample_" + str(i) + "_Q_" + Q_Signal_name + ".png"
                plotSimilarity(sims, SM_name + " between TS from " + TS_Signal_name + " Sample: " + str(i) + 
                               " and Q from " + Q_Signal_name,
                               toFileSave, GeneralPath + "Graphs/" + simFile)

                ### Plot the most likely area in the signal sample
                sim_best = min(sims)
                mostLikelyFile = "MLArea_" + SM_nameShort + "_TS_" + TS_Signal_name + "_Sample_" + str(i) + "_Q_" + Q_Signal_name + ".png"
                plotMostLikelyArea(TS_Signal, Q_Signals[0], sims, TS_Signal_name, Q_Signal_name, 
                                   "Best " + SM_name + " for " + Q_Signal_name + ": " + str(sim_best), Pad, amountPeaks,
                                  toFileSave, GeneralPath + "Graphs/" + mostLikelyFile)

                if toFileSave == 0 or toFileSave == 2:
                    print("Found best peak: ", np.argmin(sims))

                    # Spacing for visability
                    print("============================================================= ")
                    print(" ")
                    print("============================================================= ")

                # get the location of the most similar peak
                peakNum = np.argmin(sims)
                loc = MapTool.getPeaksLoc(TS_Signal, Pad, peakNum)

                # get the location of the actual peaks with the query instructions
                actualLocs = []

                for peakToLookAt in peaksToLookAt:
                    actualLoc = MapTool.getPeaksLoc(TS_Signal, Pad, peakToLookAt)
                    actualLocs.append(actualLoc)
                
                actualLocs = np.array(actualLocs)

                for actualLoc in actualLocs:
                    if loc == actualLoc:
                        correct = correct + 1

            ## This case do not graph.
            else:
                if i%10==0:
                    if toFileSave == 0 or toFileSave == 2:
                        print("Working on Sample:", i)

                # Obtain the sliding window similarity across the time series signal given the query signals
                sims = slidingWIndow_Peaks(TS_Signal, Q_Signals, MM_Scaler, Pad, amountPeaks, typeSimM, number)
                SW_Sims.append(sims)

                ### Plot the most likely area in the signal sample
                sim_best = min(sims)

                # get the location of the most similar peak
                peakNum = np.argmin(sims)
                loc = MapTool.getPeaksLoc(TS_Signal, Pad, peakNum)

                # get the location of the actual peaks with the query instructions
                actualLocs = []

                for peakToLookAt in peaksToLookAt:
                    actualLoc = MapTool.getPeaksLoc(TS_Signal, Pad, peakToLookAt)
                    actualLocs.append(actualLoc)
                
                actualLocs = np.array(actualLocs)

                for actualLoc in actualLocs:
                    if loc == actualLoc:
                        correct = correct + 1

            count = count + 1

        SW_Sims =np.array(SW_Sims)

        peakToLookAt
        
        # Note so far only works with ONE actual peak in Time Series. 
        ground_truth = np.zeros((SW_Sims.shape[0], SW_Sims.shape[1]))
        
        for actualPeak in peaksToLookAt:
            ground_truth[np.arange(SW_Sims.shape[0]), actualPeak] = 1
        

        list_of_metrics = getEvaluations(SW_Sims, ground_truth, 1000)

        accumulative_metrics = ConfidenceEstimation.calculateMultipleRoundMetrics(list_of_metrics)

        auc_Score = accumulative_metrics["auc"]

        # Gather data
        fprs = [item["fall_out"] for item in list_of_metrics]
        tprs = [item["sensitivity"] for item in list_of_metrics]

        fprs = np.array(fprs)
        tprs = np.array(tprs)

        accsSec = [item["acc"] for item in list_of_metrics]

        printOrWriteInfo(GeneralPath, "Results for Sliding Window Similiarity between the Time Series Signals " +
                         TS_Signal_name + " and the Query Signals " + Q_Signal_name, toFileSave)
        rocFile = "ROC_" + SM_nameShort + "_TS_" + TS_Signal_name + "_Q_" + Q_Signal_name
        rocTitle = "ROC for" + SM_name + "Between " + TS_Signal_name + " and " + Q_Signal_name
        aucScoreSave = int(auc_Score * 1000)

        if toFileSave == 0:
            ROC.plotROC(fprs, tprs, "red", 10, "ROC", "FP Rate", "TP Rate", rocTitle, 4, auc_Score)
        elif toFileSave == 1:
            ROC.plotROC(fprs, tprs, "red", 10, "ROC", "FP Rate", "TP Rate", rocTitle, 4, auc_Score, GeneralPath + "Graphs/" +
                       rocFile )
        elif toFileSave == 2:
            ROC.plotROC(fprs, tprs, "red", 10, "ROC", "FP Rate", "TP Rate", rocTitle, 4, auc_Score)
            ROC.plotROC(fprs, tprs, "red", 10, "ROC", "FP Rate", "TP Rate", rocTitle, 4, auc_Score, GeneralPath + "Graphs/" +
                       rocFile )
            
        printOrWriteInfo(ResultsFile, "ROC over all samples: " + str(auc_Score), toFileSave)
        printOrWriteInfo(ResultsFile, "Accuracy over all samples correctly labeled: " + str(correct/count), toFileSave)
        printOrWriteInfo(ResultsFile, "Accuracy over all labels: " + str(max(accsSec)), toFileSave)

        stop = timeit.default_timer()

        print('Time: ', stop - start)  


# In[ ]:


'''
Performs experiments with sliding window similarities between a given Time Series Signals segment and Query Signals.

Note: this is only to give general idea of how similar they are. Gives just the average similarity.
Note: the Time-Series signal segment and Queries must have the same number of peaks.

TS_Signals: (Time Series Signals) the EM signals of an execution path.
Q_Signals: (Query Signals) the EM signals of an set instruction/s.
TS_Signal_name: the name of the time series signal to be displayed in graphs
Q_Signal_name: the name of the query signal to be displayed in graphs
Pad: Used for collecting the peaks. Is the average number of features between high and low peaks.
typeSimM: The type of similarity measurment done.
    0) flipped normalized Euclidean Distance: uses euclidean distance then negatives the values. Then Min Max.
    1) Normalized Euclidean Distance.
    2) Cosine SImilarity
    3) Min-Max scale preprocessing Euclidean Distance.
peaksToLookAt: the peaks that contains the Query signal in the given Time Series Signals.
    So far only works with ONE.
amtGraphs: the amount of sliding Window similarity Graphs to show.
topNumSim: the number of sliding Window similarities to mean for one similarity. To reduce the effects of outlier Query signals.
    Note given as array to test multiple values
'''
def averageSimilarityExp(TS_Signals, Q_Signals, TS_Signal_name, Q_Signal_name, Pad, peaksToLookAt, typeSimM,
                     amtGraphs = 20, topNumSim=np.array([10]), toFileSave = 0, GeneralPath= ""):
    
    if toFileSave < 0 or toFileSave > 2:
        print("Error: non valid toFileSave value.")
        
    if toFileSave == 1 or toFileSave == 2:
        if GeneralPath == "":
            print("Error: no path specified.")
    
    ResultsFile = GeneralPath + "/results.txt"
    
    ## Create path if does not exist.
    # Check whether the specified path exists or not
    isExist = os.path.exists(GeneralPath)

    if not isExist:
      # Create a new directory because it does not exist 
      os.makedirs(GeneralPath)
    
    # Get the number of peaks in the Query Signals
    amountPeaks = MapTool.getPeaks(Q_Signals[0], Pad).shape[0]
    amountPeaksTS = MapTool.getPeaks(TS_Signals[0], Pad).shape[0]
    amountPeaksQ = MapTool.getPeaks(Q_Signals[0], Pad).shape[0]
    
    if amountPeaksTS != amountPeaksQ:
        print("Error: differing numnber of peaks.")
    
    # Create the Min-Max Scalar
    MM_Scaler = MinMaxScaler()
    
    # String name of the similarity measurement.
    SM_name = ""
    if typeSimM == 0:
        SM_name = "flipped normalized Euclidean Distance"
        SM_nameShort = "FNED"
    elif typeSimM == 1:
        SM_name = "Normalized Euclidean Distance"
        SM_nameShort = "NED"
    elif typeSimM == 2:
        SM_name = "Cosine Similarity"
        SM_nameShort = "CS"
    elif typeSimM == 3:
        SM_name = "MinMax Euclidean Distance"
        SM_nameShort = "MED"
        
    ResultsFile = GeneralPath + SM_nameShort + "_TS_" + TS_Signal_name + "_Q_" + Q_Signal_name + " results.txt"
    
    printOrWriteInfo(ResultsFile, "Beginning Tests...", toFileSave, True)
    
    # check for errors:
    checkCount = []
    for Q_Signal in Q_Signals:
        checkCount.append(MapTool.getPeaks(Q_Signal, Pad).shape[0])
    
    checkCount = np.array(checkCount)
    checkCount = np.unique(checkCount)
    
    if checkCount.shape[0] > 1:
        printOrWriteInfo(ResultsFile, "Error in gathering Q_Signals as unique peaks gave different values", toFileSave)
        printOrWriteInfo(ResultsFile, checkCount, toFileSave)
    
    
    # Create a Min Max Scaler for later use with Min-Max scale preprocessing Euclidean Distance
    if typeSimM == 3:
        # gather all the peaks of the Time Series Signals and Query Signals
        if nonPeakEnds:
            # Obtain the section of the TS_Signal to do similarity.
            k = i+amountPeaks-2
        else:
            k = i+amountPeaks
            
        TS_Signals_Peaks = MapTool.getPeaksDataset(TS_Signals, Pad)
        TS_Signals_Peaks = TS_Signals_Peaks[:][:k]
            
        Q_Signals_Peaks =  MapTool.getPeaksDataset(Q_Signals, Pad)
        Q_Signals_Peaks = Q_Signals_Peaks[:][:k]
        
        print(TS_Signals_Peaks[0])
        plt.plot(TS_Signals_Peaks[0])
        plt.show()
        
        print(Q_Signals_Peaks[0])
        plt.plot(Q_Signals_Peaks[0])
        plt.show()
        
        # Find the max and min of the two.
        TS_Max = np.max(TS_Signals_Peaks)
        Q_Max = np.max(Q_Signals_Peaks)
        TS_Min = np.min(TS_Signals_Peaks)
        Q_Min = np.min(Q_Signals_Peaks)
        
        # place into array
        info = np.array([TS_Max, Q_Max, TS_Min, Q_Min])
        info = np.reshape(info, (-1,1)) 
        
        # Create the Min-Max Scalar based on the min max of the info collected.
        MM_Scaler.fit(info)
    
    # Set the number of the top number of sliding Window similarities.
    ## this will go through each to test each number given
    for number in topNumSim:
        start = timeit.default_timer()
    
        count = 0
        correct =  0
        SW_Sims = []
        
        
        printOrWriteInfo(ResultsFile, " ", toFileSave)
        printOrWriteInfo(ResultsFile, "Taking the top number of sliding Window similarities: " + str(number), toFileSave)
        
        # Get a Time Series Signal for similarity measurement
        for i in range(TS_Signals.shape[0]):

            TS_Signal = TS_Signals[i]

            # Find out if to graph the sliding window similarity measurement for the time series signal
            ## This case graph
            if i < amtGraphs:
                # Title of section
                if toFileSave == 0 or toFileSave == 2:
                    print(TS_Signal_name + ": Signal Sample " + str(i))

                # Obtain the sliding window similarity across the time series signal given the query signals
                sims = slidingWIndow_Peaks(TS_Signal, Q_Signals, MM_Scaler, Pad, amountPeaks, typeSimM, number)
                if sims.shape != 1:
                    print("Error: more than one comparison done for this similarity.")
                    print("Similarities calculated: " + str(sims.shape))
                SW_Sims.append(sims)

                ### Plot the correlation
                simFile = "Sim_" + SM_nameShort + "_TS_" + TS_Signal_name + "_Sample_" + str(i) + "_Q_" + Q_Signal_name + ".png"
                plotSimilarity(sims, SM_name + " between TS from " + TS_Signal_name + " Sample: " + str(i) + 
                               " and Q from " + Q_Signal_name,
                               toFileSave, GeneralPath + "Graphs/" + simFile)

                ### Plot the most likely area in the signal sample
                sim_best = min(sims)
                mostLikelyFile = "MLArea_" + SM_nameShort + "_TS_" + TS_Signal_name + "_Sample_" + str(i) + "_Q_" + Q_Signal_name + ".png"
                plotMostLikelyArea(TS_Signal, Q_Signals[0], sims, TS_Signal_name, Q_Signal_name, 
                                   "Best " + SM_name + " for " + Q_Signal_name + ": " + str(sim_best), Pad, amountPeaks,
                                  toFileSave, GeneralPath + "Graphs/" + mostLikelyFile)

                if toFileSave == 0 or toFileSave == 2:
                    print("Found best peak: ", np.argmin(sims))

                    # Spacing for visability
                    print("============================================================= ")
                    print(" ")
                    print("============================================================= ")

                # get the location of the most similar peak
                peakNum = np.argmin(sims)
                loc = MapTool.getPeaksLoc(TS_Signal, Pad, peakNum)

                # get the location of the actual peaks with the query instructions
                actualLocs = []

                for peakToLookAt in peaksToLookAt:
                    actualLoc = MapTool.getPeaksLoc(TS_Signal, Pad, peakToLookAt)
                    actualLocs.append(actualLoc)
                
                actualLocs = np.array(actualLocs)

                for actualLoc in actualLocs:
                    if loc == actualLoc:
                        correct = correct + 1

            ## This case do not graph.
            else:
                if i%10==0:
                    if toFileSave == 0 or toFileSave == 2:
                        print("Working on Sample:", i)

                # Obtain the sliding window similarity across the time series signal given the query signals
                sims = slidingWIndow_Peaks(TS_Signal, Q_Signals, MM_Scaler, Pad, amountPeaks, typeSimM, number)
                SW_Sims.append(sims)

                ### Plot the most likely area in the signal sample
                sim_best = min(sims)

                # get the location of the most similar peak
                peakNum = np.argmin(sims)
                loc = MapTool.getPeaksLoc(TS_Signal, Pad, peakNum)

                # get the location of the actual peaks with the query instructions
                actualLocs = []

                for peakToLookAt in peaksToLookAt:
                    actualLoc = MapTool.getPeaksLoc(TS_Signal, Pad, peakToLookAt)
                    actualLocs.append(actualLoc)
                
                actualLocs = np.array(actualLocs)

                for actualLoc in actualLocs:
                    if loc == actualLoc:
                        correct = correct + 1

            count = count + 1

        SW_Sims =np.array(SW_Sims)
            
        printOrWriteInfo(ResultsFile, "Average Similarity: " + str(np.mean(SW_Sims)), toFileSave)

        stop = timeit.default_timer()

        print('Time: ', stop - start)  


# In[2]:


'''
Performs experiments with sliding window similarities between a given Time Series Signals segment and Query Signals.

Note: this is only to give general idea of how similar they are. Gives just the average similarity.
Note: the Time-Series signal segment and Queries must have the same number of peaks.

TS_Signals: (Time Series Signals) the EM signals of an execution path.
Q_Signals: (Query Signals) the EM signals of an set instruction/s.
TS_Signal_name: the name of the time series signal to be displayed in graphs
Q_Signal_name: the name of the query signal to be displayed in graphs
Pad: Used for collecting the peaks. Is the average number of features between high and low peaks.
typeSimM: The type of similarity measurment done.
    0) flipped normalized Euclidean Distance: uses euclidean distance then negatives the values. Then Min Max.
    1) Normalized Euclidean Distance.
    2) Cosine SImilarity
    3) Min-Max scale preprocessing Euclidean Distance.
peaksToLookAt: the peaks that contains the Query signal in the given Time Series Signals.
Remove_Peak_List: Peaks to remove before comparing.
amtGraphs: the amount of sliding Window similarity Graphs to show.
topNumSim: the number of sliding Window similarities to mean for one similarity. To reduce the effects of outlier Query signals.
    Note given as array to test multiple values
'''
def averageSimilarityExpReturn(
    TS_Signals, Q_Signals, TS_Signal_name, Q_Signal_name, Pad, peaksToLookAt, typeSimM,
    Remove_Peak_List=[], amtGraphs=20, topNumSim=np.array([10]), toFileSave=0,
    GeneralPath="", nonPeakEnds=False, Info=True, PeakOnlyData=False, removeLastPeak=True
):
    import os
    import numpy as np
    import time
    from sklearn.preprocessing import MinMaxScaler

    if toFileSave not in [0, 1, 2]:
        raise ValueError("Invalid toFileSave value.")
    if toFileSave > 0 and not GeneralPath:
        raise ValueError("GeneralPath must be specified when saving files.")

    if toFileSave > 0:
        os.makedirs(GeneralPath, exist_ok=True)
        results_file_path = os.path.join(GeneralPath, "results.txt")

    # Helper to log or print info
    def log_info(message):
        if Info:
            print(message)
        if toFileSave > 0:
            with open(results_file_path, "a") as f:
                f.write(message + "\n")
    
    # Precompute peak data if needed
    if typeSimM != 6:
        if not PeakOnlyData:
            TS_Signals_Peaks = MapTool.getPeaksDataset(TS_Signals, Pad)
            Q_Signals_Peaks = MapTool.getPeaksDataset(Q_Signals, Pad)
            if removeLastPeak:
                TS_Signals_Peaks = np.array(TS_Signals_Peaks[:,:len(TS_Signals_Peaks[0])-1])
                Q_Signals_Peaks = np.array(Q_Signals_Peaks[:,:len(Q_Signals_Peaks[0])-1])
                print(TS_Signals_Peaks.shape)
                print(Q_Signals_Peaks.shape)
        else:
            TS_Signals_Peaks = TS_Signals
            Q_Signals_Peaks = Q_Signals

        # Validate peak counts
        amountPeaksQ = Q_Signals_Peaks[0].shape[0]
        amountPeaksTS = TS_Signals_Peaks[0].shape[0]

        if amountPeaksQ != amountPeaksTS:
            raise ValueError("Query and Time Series signals must have the same number of peaks.")

    # Min-Max Scaler setup
    if typeSimM == 3:
        combined_data = np.vstack((TS_Signals_Peaks, Q_Signals_Peaks))
        MM_Scaler = MinMaxScaler()
        MM_Scaler.fit(combined_data)

    # Predefine result storage
    results = []

    for num_top in topNumSim:
        start_time = time.time()
        log_info(f"Processing topNumSim={num_top}...")

        similarities = []
        if typeSimM == 6:
            for i, TS_Signal in enumerate(TS_Signals):
                if i % 10 == 0:
                    log_info(f"Processing sample {i}/{len(TS_Signals)}...")
                
                amountPeaksQ=0 
                sims = slidingWIndow_Peaks(
                    TS_Signal, Q_Signals, MM_Scaler if typeSimM == 3 else None,
                    Pad, amountPeaksQ, typeSimM, num_top, Remove_Peak_List, nonPeakEnds, PeakOnlyData=True
                )
                similarities.append(np.mean(sims))

                if i < amtGraphs:
                    log_info(f"Plotting for sample {i}...")
                    # Generate and save graphs here if needed
                    
        else:
            for i, TS_Signal_Peaks in enumerate(TS_Signals_Peaks):
                if i % 10 == 0:
                    log_info(f"Processing sample {i}/{len(TS_Signals)}...")
   
                sims = slidingWIndow_Peaks(
                    TS_Signal_Peaks, Q_Signals_Peaks, MM_Scaler if typeSimM == 3 else None,
                    Pad, amountPeaksQ, typeSimM, num_top, Remove_Peak_List, nonPeakEnds, PeakOnlyData=True
                )
                similarities.append(np.mean(sims))

                if i < amtGraphs:
                    log_info(f"Plotting for sample {i}...")
                    # Generate and save graphs here if needed

        avg_similarity = np.mean(similarities)
        log_info(f"Average Similarity for topNumSim={num_top}: {avg_similarity:.4f}")

        results.append(avg_similarity)
        log_info(f"Time taken for topNumSim={num_top}: {time.time() - start_time:.2f} seconds")

    return np.array(results), np.array(similarities)

'''
Old Method 1_15_2025

Retired due to slow speed and redundency


def averageSimilarityExpReturn(TS_Signals, Q_Signals, TS_Signal_name, Q_Signal_name, Pad, peaksToLookAt, typeSimM,
                                      Remove_Peak_List =[],
                     amtGraphs = 20, topNumSim=np.array([10]), toFileSave = 0, GeneralPath= "", nonPeakEnds = False, Info = True,
                              PeakOnlyData = False):
    
    if toFileSave < 0 or toFileSave > 2:
        print("Error: non valid toFileSave value.")
        
    if toFileSave == 1 or toFileSave == 2:
        if GeneralPath == "":
            print("Error: no path specified.")
    
    ResultsFile = GeneralPath + "/results.txt"
    
    if toFileSave == 1 or toFileSave == 2:
        ## Create path if does not exist.
        # Check whether the specified path exists or not
        isExist = os.path.exists(GeneralPath)

        if not isExist:
          # Create a new directory because it does not exist 
          os.makedirs(GeneralPath)
    
    # Get the number of peaks in the Query Signals
    if PeakOnlyData:
        amountPeaks = Q_Signals[0].shape[0]
        amountPeaksTS = TS_Signals[0].shape[0]
        amountPeaksQ = Q_Signals[0].shape[0]
    else:
        amountPeaks = MapTool.getPeaks(Q_Signals[0], Pad).shape[0]
        amountPeaksTS = MapTool.getPeaks(TS_Signals[0], Pad).shape[0]
        amountPeaksQ = MapTool.getPeaks(Q_Signals[0], Pad).shape[0]

    if amountPeaksTS != amountPeaksQ:
        print("Error: differing numnber of peaks.")
    
    # Create the Min-Max Scalar
    MM_Scaler = MinMaxScaler()
    
    # String name of the similarity measurement.
    SM_name = ""
    if typeSimM == 0:
        SM_name = "flipped normalized Euclidean Distance"
        SM_nameShort = "FNED"
    elif typeSimM == 1:
        SM_name = "Normalized Euclidean Distance"
        SM_nameShort = "NED"
    elif typeSimM == 2:
        SM_name = "Cosine Similarity"
        SM_nameShort = "CS"
    elif typeSimM == 3:
        SM_name = "MinMax Euclidean Distance"
        SM_nameShort = "MED"
    elif typeSimM == 4:
        SM_name = "Euclidean Distance"
        SM_nameShort = "ED"
        
    ResultsFile = GeneralPath + SM_nameShort + "_TS_" + TS_Signal_name + "_Q_" + Q_Signal_name + " results.txt"
    
    if Info:
        printOrWriteInfo(ResultsFile, "Beginning Tests...", toFileSave, True)
    
    # check for errors:
    if PeakOnlyData == False:
        checkCount = []
        for Q_Signal in Q_Signals:
            checkCount.append(MapTool.getPeaks(Q_Signal, Pad).shape[0])

        checkCount = np.array(checkCount)
        checkCount = np.unique(checkCount)

        if checkCount.shape[0] > 1:
            printOrWriteInfo(ResultsFile, "Error in gathering Q_Signals as unique peaks gave different values", toFileSave)
            printOrWriteInfo(ResultsFile, checkCount, toFileSave)
    
    
    # Create a Min Max Scaler for later use with Min-Max scale preprocessing Euclidean Distance
    if typeSimM == 3:
        # gather all the peaks of the Time Series Signals and Query Signals
        if nonPeakEnds:
            # Obtain the section of the TS_Signal to do similarity.
            k = amountPeaks-1
        else:
            k = amountPeaks
        
        if PeakOnlyData:
            TS_Signals_Peaks = TS_Signals
            Q_Signals_Peaks = Q_Signals
        else:   
            TS_Signals_Peaks = MapTool.getPeaksDataset(TS_Signals, Pad)
            Q_Signals_Peaks =  MapTool.getPeaksDataset(Q_Signals, Pad)
        
        TS_Signals_Peaks = TS_Signals_Peaks[:,:k]
        Q_Signals_Peaks = Q_Signals_Peaks[:,:k]
        
        # Find the max and min of the two.
        TS_Max = np.max(TS_Signals_Peaks)
        Q_Max = np.max(Q_Signals_Peaks)
        TS_Min = np.min(TS_Signals_Peaks)
        Q_Min = np.min(Q_Signals_Peaks)
        
        # place into array
        info = np.array([TS_Max, Q_Max, TS_Min, Q_Min])
        info = np.reshape(info, (-1,1)) 
        
        # Create the Min-Max Scalar based on the min max of the info collected.
        MM_Scaler.fit(info)
    
    # Set the number of the top number of sliding Window similarities.
    ## this will go through each to test each number given
    for number in topNumSim:
        start = timeit.default_timer()
    
        count = 0
        correct =  0
        SW_Sims = []
        
        if Info:
            printOrWriteInfo(ResultsFile, " ", toFileSave)
            printOrWriteInfo(ResultsFile, "Taking the top number of sliding Window similarities: " + str(number), toFileSave)
        
        # Get a Time Series Signal for similarity measurement
        for i in range(TS_Signals.shape[0]):

            TS_Signal = TS_Signals[i]

            # Find out if to graph the sliding window similarity measurement for the time series signal
            ## This case graph
            if i < amtGraphs:
                # Title of section
                if toFileSave == 0 or toFileSave == 2:
                    print(TS_Signal_name + ": Signal Sample " + str(i))

                # Obtain the sliding window similarity across the time series signal given the query signals
                sims = slidingWIndow_Peaks(TS_Signal, Q_Signals, MM_Scaler, Pad, amountPeaks, typeSimM, number, Remove_Peak_List,
                                           nonPeakEnds, PeakOnlyData)
                if sims.shape != 1:
                    print("Error: more than one comparison done for this similarity.")
                    print("Similarities calculated: " + str(sims.shape))
                SW_Sims.append(sims)

                ### Plot the correlation
                simFile = "Sim_" + SM_nameShort + "_TS_" + TS_Signal_name + "_Sample_" + str(i) + "_Q_" + Q_Signal_name + ".png"
                plotSimilarity(sims, SM_name + " between TS from " + TS_Signal_name + " Sample: " + str(i) + 
                               " and Q from " + Q_Signal_name,
                               toFileSave, GeneralPath + "Graphs/" + simFile)

                ### Plot the most likely area in the signal sample
                sim_best = min(sims)
                mostLikelyFile = "MLArea_" + SM_nameShort + "_TS_" + TS_Signal_name + "_Sample_" + str(i) + "_Q_" + Q_Signal_name + ".png"
                plotMostLikelyArea(TS_Signal, Q_Signals[0], sims, TS_Signal_name, Q_Signal_name, 
                                   "Best " + SM_name + " for " + Q_Signal_name + ": " + str(sim_best), Pad, amountPeaks,
                                  toFileSave, GeneralPath + "Graphs/" + mostLikelyFile)

                if toFileSave == 0 or toFileSave == 2:
                    print("Found best peak: ", np.argmin(sims))

                    # Spacing for visability
                    print("============================================================= ")
                    print(" ")
                    print("============================================================= ")

                # get the location of the most similar peak
                peakNum = np.argmin(sims)
                if PeakOnlyData:
                    loc = peakNum
                else:
                    loc = MapTool.getPeaksLoc(TS_Signal, Pad, peakNum)

                # get the location of the actual peaks with the query instructions
                actualLocs = []

                for peakToLookAt in peaksToLookAt:
                    if PeakOnlyData:
                        actualLoc  = peakToLookAt
                    else:
                        actualLoc = MapTool.getPeaksLoc(TS_Signal, Pad, peakToLookAt)
                    actualLocs.append(actualLoc)
                
                actualLocs = np.array(actualLocs)

                for actualLoc in actualLocs:
                    if loc == actualLoc:
                        correct = correct + 1

            ## This case do not graph.
            else:
                if i%10==0:
                    if toFileSave == 0 or toFileSave == 2:
                        if Info:
                            print("Working on Sample:", i)

                # Obtain the sliding window similarity across the time series signal given the query signals
                sims = slidingWIndow_Peaks(TS_Signal, Q_Signals, MM_Scaler, Pad, amountPeaks, typeSimM, number, Remove_Peak_List,
                                           nonPeakEnds, PeakOnlyData)
                SW_Sims.append(sims)

                ### Plot the most likely area in the signal sample
                sim_best = min(sims)

                # get the location of the most similar peak
                peakNum = np.argmin(sims)
                if PeakOnlyData:
                    loc = peakNum
                else:
                    loc = MapTool.getPeaksLoc(TS_Signal, Pad, peakNum)

                # get the location of the actual peaks with the query instructions
                actualLocs = []

                for peakToLookAt in peaksToLookAt:
                    if PeakOnlyData:
                        actualLoc  = peakToLookAt
                    else:
                        actualLoc = MapTool.getPeaksLoc(TS_Signal, Pad, peakToLookAt)
                    actualLocs.append(actualLoc)
                
                actualLocs = np.array(actualLocs)

                for actualLoc in actualLocs:
                    if loc == actualLoc:
                        correct = correct + 1

            count = count + 1

        SW_Sims =np.array(SW_Sims)
        
        if Info:
            printOrWriteInfo(ResultsFile, "Average Similarity: " + str(np.mean(SW_Sims)), toFileSave)

        stop = timeit.default_timer()

        if Info:
            print('Time: ', stop - start)  
        
        return SW_Sims
'''


# In[1]:


def corrPeakExp(Signals, Seqs, Signal_name, Seq_name, pad, peakToLookAt, corr_scale= -100, amtGraphs = 20):
    start = timeit.default_timer()
    
    count = 0
    correct =  0
    euc_corrs = []
    
    # check for errors:
    checkCount = []
    for seq in Seqs:
        checkCount.append(MapTool.getPeaks(seq, pad).shape[0])
    
    checkCount = np.array(checkCount)
    checkCount = np.unique(checkCount)
    
    if checkCount.shape[0] > 1:
        print("Error in gathering seqs as unique peaks gave different values")
        print(checkCount)
    
    
    amountPeaks = MapTool.getPeaks(Seqs[0], pad).shape[0]

    for i in range(Signals.shape[0]):

        Signal = Signals[i]

        if i < amtGraphs:
            # Title of section
            print(Signal_name + ": Signal Sample ", i)

            euc_corr = rollingEuclidean_Peaks(Signal, Seqs, corr_scale, pad, amountPeaks)
            euc_corrs.append(euc_corr)

            ### Plot the correlation
            plotCorrelation(euc_corr, "Correlation between " + Signal_name + " Sample: " + str(i) + " and " + Seq_name)

            ### Plot the most likely area in the signal sample
            corr_best = max(euc_corr)
            plotMostLikelyArea(Signal, Seqs[0], euc_corr, Signal_name, Seq_name, 
                               "Best Correlation for " + Seq_name + ", Corr is: " + str(corr_best), pad)

            print("Found best peak: ", np.argmax(euc_corr))

            # Spacing for visability
            print("============================================================= ")
            print(" ")
            print("============================================================= ")

            peakNum = np.argmax(euc_corr)
            loc = MapTool.getPeaksLoc(Signal, pad, peakNum)

            actualLocs = peakToLookAt

            actualLoc = MapTool.getPeaksLoc(Signal, pad, actualLocs)

            if loc == actualLoc:
                correct = correct + 1

        else:
            if i%10==0:
                print("Working on Sample:", i)

            ### Plot the correlation
            euc_corr = rollingEuclidean_Peaks(Signal, Seqs, corr_scale, pad, amountPeaks)
            
            euc_corrs.append(euc_corr)

            ### Plot the most likely area in the signal sample

            peakNum = np.argmax(euc_corr)
            loc = MapTool.getPeaksLoc(Signal, pad, peakNum)

            actualLocs = peakToLookAt

            actualLoc = MapTool.getPeaksLoc(Signal, pad, actualLocs)

            if loc == actualLoc:
                correct = correct + 1

        count = count + 1

    euc_corrs =np.array(euc_corrs)

    ground_truth = np.zeros((euc_corrs.shape[0], euc_corrs.shape[1]))
    ground_truth[np.arange(euc_corrs.shape[0]),peakToLookAt] = 1

    list_of_metrics = getEvaluations(euc_corrs, ground_truth, 1000)

    accumulative_metrics = ConfidenceEstimation.calculateMultipleRoundMetrics(list_of_metrics)

    auc_Score = accumulative_metrics["auc"]

    # Gather data
    fprs = [item["fall_out"] for item in list_of_metrics]
    tprs = [item["sensitivity"] for item in list_of_metrics]

    fprs = np.array(fprs)
    tprs = np.array(tprs)

    accsSec = [item["acc"] for item in list_of_metrics]

    # f1 and Acc
    ''' 
    # Other data if necessary
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
    '''
    
    print("Results for window euclidean corralation between the Signals " + Signal_name + " and the instruction Sequences " + Seq_name)
    rocTitle = "ROC graph for all euclidean correlations."
    aucScoreSave = int(auc_Score * 1000)

    ROC.plotROC(fprs, tprs, "red", 10, "ROC", "FP Rate", "TP Rate", rocTitle, 4, auc_Score)

    print("Accuracy over all samples correctly labeled: ", correct/count)
    print("Accuracy over all labels: ", max(accsSec))
    
    stop = timeit.default_timer()
    
    print('Time: ', stop - start)  


# In[ ]:


'''
Tests various number of top corr comparisons for each peak correlation sequence to account for.
Produces an Accuracy for each number from 5 to 50 skiping by 5.

Does this by sum and adv. may remove sum for adv only later as this keeps in the 0-1 range for AUC gathering.
'''
def corrPeakExpTestNumCorr(Signals, Seqs, Signal_name, Seq_name, pad, peakToLookAt, corr_scale= -100, amtGraphs = 20, topNumCorr=50):
    start = timeit.default_timer()
    
    count = 0
    correct =  0
    
    # check for errors:
    checkCount = []
    for seq in Seqs:
        checkCount.append(MapTool.getPeaks(seq, pad).shape[0])
    
    checkCount = np.array(checkCount)
    checkCount = np.unique(checkCount)
    
    if checkCount.shape[0] > 1:
        print("Error in gathering seqs as unique peaks gave different values")
        print(checkCount)
    
    
    amountPeaks = MapTool.getPeaks(Seqs[0], pad).shape[0]

    for i in range(amtGraphs):

        Signal = Signals[i]

        # Title of section
        print(Signal_name + ": Signal Sample ", i)

        for number in range(0,topNumCorr, 5):
            if number == 0:
                number = 1

            print(" ")
            print("Taking the top number of correlations: ", number)

            euc_corr = rollingEuclidean_PeaksTest(Signal, Seqs, corr_scale, pad, amountPeaks, number, False)

            ### Plot the correlation
            # plotCorrelation(euc_corr, "Correlation between " + Signal_name + " Sample: " + str(i) + " and " + Seq_name)

            ### Plot the most likely area in the signal sample
            corr_best = max(euc_corr)
            # plotMostLikelyArea(Signal, Seqs[0], euc_corr, Signal_name, Seq_name, 
            #                   "Best Correlation for " + Seq_name + ", Corr is: " + str(corr_best), pad)

            print("Best Correlation: ", corr_best)
            print("Found best peak: ", np.argmax(euc_corr))

            # Accuracy by signal.
            euc_corr = np.array(euc_corr)

            ground_truth = np.zeros(euc_corr.shape)
            ground_truth[peakToLookAt] = 1

            list_of_metrics = getEvaluationsOne(euc_corr, ground_truth, 1000)

            accumulative_metrics = ConfidenceEstimation.calculateMultipleRoundMetrics(list_of_metrics)

            auc_Score = accumulative_metrics["auc"]

            # Gather data
            fprs = [item["fall_out"] for item in list_of_metrics]
            tprs = [item["sensitivity"] for item in list_of_metrics]

            fprs = np.array(fprs)
            tprs = np.array(tprs)

            accsSec = [item["acc"] for item in list_of_metrics]


            #print("Results for window euclidean corralation between the Signals " + Signal_name + " and the instruction Sequences " + Seq_name)
            #rocTitle = "ROC graph for all euclidean correlations."
            #aucScoreSave = int(auc_Score * 1000)

            #ROC.plotROC(fprs, tprs, "red", 10, "ROC", "FP Rate", "TP Rate", rocTitle, 4, auc_Score)

            print("Accuracy over all labels: ", max(accsSec))

            stop = timeit.default_timer()  


# In[ ]:


'''
Tests various number of top corr comparisons for each peak correlation sequence to account for.
Produces an Accuracy for each number from 5 to 50 skiping by 5.

Does this by sum and adv. may remove sum for adv only later as this keeps in the 0-1 range for AUC gathering.
'''
def corrPeakExpNumCorr(Signals, Seqs, Signal_name, Seq_name, pad, peakToLookAt, corr_scale= -100, amtGraphs = 20, topNumCorr=np.array([10])):
    
    # check for errors:
    checkCount = []
    for seq in Seqs:
        checkCount.append(MapTool.getPeaks(seq, pad).shape[0])
    
    checkCount = np.array(checkCount)
    checkCount = np.unique(checkCount)
    
    if checkCount.shape[0] > 1:
        print("Error in gathering seqs as unique peaks gave different values")
        print(checkCount)
    
    
    amountPeaks = MapTool.getPeaks(Seqs[0], pad).shape[0]
    
    for number in topNumCorr:
        start = timeit.default_timer()
    
        count = 0
        correct =  0
        euc_corrs = []
        
        
        print(" ")
        print("Taking the top number of correlations: ", number)
        
        for i in range(Signals.shape[0]):

            Signal = Signals[i]

            if i < amtGraphs:
                # Title of section
                print(Signal_name + ": Signal Sample ", i)

                euc_corr = rollingEuclidean_Peaks(Signal, Seqs, corr_scale, pad, amountPeaks, number)
                euc_corrs.append(euc_corr)

                ### Plot the correlation
                plotCorrelation(euc_corr, "Correlation between " + Signal_name + " Sample: " + str(i) + " and " + Seq_name)

                ### Plot the most likely area in the signal sample
                corr_best = max(euc_corr)
                plotMostLikelyArea(Signal, Seqs[0], euc_corr, Signal_name, Seq_name, 
                                   "Best Correlation for " + Seq_name + ", Corr is: " + str(corr_best), pad)

                print("Found best peak: ", np.argmax(euc_corr))

                # Spacing for visability
                print("============================================================= ")
                print(" ")
                print("============================================================= ")

                peakNum = np.argmax(euc_corr)
                loc = MapTool.getPeaksLoc(Signal, pad, peakNum)

                actualLocs = peakToLookAt

                actualLoc = MapTool.getPeaksLoc(Signal, pad, actualLocs)

                if loc == actualLoc:
                    correct = correct + 1

            else:
                if i%10==0:
                    print("Working on Sample:", i)

                ### Plot the correlation
                euc_corr = rollingEuclidean_Peaks(Signal, Seqs, corr_scale, pad, amountPeaks, number)

                euc_corrs.append(euc_corr)

                ### Plot the most likely area in the signal sample

                peakNum = np.argmax(euc_corr)
                loc = MapTool.getPeaksLoc(Signal, pad, peakNum)

                actualLocs = peakToLookAt

                actualLoc = MapTool.getPeaksLoc(Signal, pad, actualLocs)

                if loc == actualLoc:
                    correct = correct + 1

            count = count + 1

        euc_corrs =np.array(euc_corrs)

        ground_truth = np.zeros((euc_corrs.shape[0], euc_corrs.shape[1]))
        ground_truth[np.arange(euc_corrs.shape[0]),peakToLookAt] = 1

        list_of_metrics = getEvaluations(euc_corrs, ground_truth, 1000)

        accumulative_metrics = ConfidenceEstimation.calculateMultipleRoundMetrics(list_of_metrics)

        auc_Score = accumulative_metrics["auc"]

        # Gather data
        fprs = [item["fall_out"] for item in list_of_metrics]
        tprs = [item["sensitivity"] for item in list_of_metrics]

        fprs = np.array(fprs)
        tprs = np.array(tprs)

        accsSec = [item["acc"] for item in list_of_metrics]

        print("Results for window euclidean corralation between the Signals " + Signal_name + " and the instruction Sequences " + Seq_name)
        rocTitle = "ROC graph for all euclidean correlations."
        aucScoreSave = int(auc_Score * 1000)

        ROC.plotROC(fprs, tprs, "red", 10, "ROC", "FP Rate", "TP Rate", rocTitle, 4, auc_Score)

        print("Accuracy over all samples correctly labeled: ", correct/count)
        print("Accuracy over all labels: ", max(accsSec))

        stop = timeit.default_timer()

        print('Time: ', stop - start)  


        
# In[ ]:

import os

def checkAndMakeFolder(path):
    # Check whether the specified path exists or not
    isExist = os.path.exists(path)
    if not isExist:
       # Create a new directory because it does not exist
       os.makedirs(path)
        
# In[ ]:

def printandwrite(loc, text, firstLine = False, printIt=True):       
    if printIt:
        print(text)
    if firstLine:
        sourceFile = open(loc, 'w')
    else:
        sourceFile = open(loc, 'a')
    sourceFile.write(text)
    sourceFile.close()
        
# In[ ]:

'''
Removes the prior and after information by having a start and end to the mid point.
'''
def removePriorandAfterInfo(cycle, midpoint=0):
    start = 0
    end = len(cycle)-1
    
    #plt.plot(cycle)
    #plt.show()
    #plt.close()
    
    #print(cycle)
    
    stop = 0
    while stop == 0:
        if cycle[start] < midpoint:
            start = start+1
        else:
            stop = 1
    # next need to remove the end of the signal information of the next instruction.
    stop = 0
    while stop == 0:
        if cycle[end] > midpoint:
            end = end-1
        else:
            stop = 1
    
    end = end +1
    
    #plt.plot(cycle[start:end])
    #plt.show()
    #plt.close()
    
    #print(cycle[start:end])
    
    return cycle[start:end]

# In[ ]:

'''
Obtains the cycles of the signals. 

INPUT:
    dataset: array of signals.
    pad:the average number of features between high and low peaks.
    
RETURNS:
    cycle_Dataset: list of list of the cycles for each signal.
'''
def getCyclesDataset(dataset, pad):
    cycle_Dataset = []
    for signal in dataset:
        cycles = []
        start = 0
        end = 2
        numberOfCycles = (len(MapTool.getPeaks(signal,pad))-1)/2
        #print(numberOfCycles)
        
        while start < numberOfCycles*2:
            #print("Start: ", start)
            #print("End: ", end)
            if start==0:
                cycle = signal[:MapTool.getPeaksLoc(signal, pad, end)]
                #plt.plot(cycle)
                cycle = removePriorandAfterInfo(cycle)
                #plt.plot(cycle)
                cycles.append(cycle)
                start = end
                end = end +2
            else:
                cycle = signal[MapTool.getPeaksLoc(signal, pad, start, getHigh = False):MapTool.getPeaksLoc(signal, pad, end)]
                #plt.plot(cycle)
                cycle = removePriorandAfterInfo(cycle)
                #plt.plot(cycle)
                cycles.append(cycle)
                start = end
                end = end+2
                
        cycle_Dataset.append(cycles)
        
    return np.array(cycle_Dataset, dtype=object)

# In[ ]:

'''
Obtain the sum of CC across the whole signal for each signal in TS_dataset.
INPUT:
    CyclesCCs: Cycles average cross correlation
    
RETURN:
    CCs: added cross correlation for the whole signal for each signals.
'''
def getTotalCC(CyclesCCs):
    CCs = np.sum(CyclesCCs, axis=0)
    return CCs

# In[]:

def normalized_cross_correlation(x, y):
    return sciSignal.correlate(x, y, mode='full', method='fft') / (np.linalg.norm(x) * np.linalg.norm(y))

# In[ ]:

from scipy import signal as sciSignal

'''
Obtains the average of the top Cross-correlations for each cycle
INPUT: 
    TS_Signals: (Time Series Signals) the EM signals of an execution path.
    Q_Signals: (Query Signals) the EM signals of an set instruction/s.
    Pad: Used for collecting the peaks. Is the average number of features between high and low peaks.
    numCC: the number of cross correlations to average. 

RETURNS:
    CycleCC: list of lists of average correlations.
    CCbyCycle: lists of each signals avg cross correlation added to each other.
'''
def cross_Correlation_by_Cycles(TS_Signals, Q_Signals, pad, numCC, path, printIt=True):
    start_time = time.time()
    # Obtain the cycles of each dataset
    TS_Cycles = getCyclesDataset(TS_Signals, pad)  # Shape: [num_TS, num_cycles]
    Q_Cycles = getCyclesDataset(Q_Signals, pad)    # Shape: [num_Q, num_cycles]

    num_cycles = TS_Cycles.shape[1]
    CyclesCCs = []

    for i in range(num_cycles):
        # Extract the i-th cycle for all TS and Q signals
        TS_cycle_signals = TS_Cycles[:, i]  # Shape: [num_TS]
        Q_cycle_signals = Q_Cycles[:, i]    # Shape: [num_Q]

        CC_for_Cycle = []

        # Compute cross-correlations for each TS signal against all Q signals
        for Q_Cycle in Q_cycle_signals:
            # Vectorized correlation calculation using list comprehension
            correlations = np.array([
                np.max(normalized_cross_correlation(TS_Cycle, Q_Cycle))
                for TS_Cycle in TS_cycle_signals
            ])

            # Efficient top-k selection
            top_indices = np.argpartition(correlations, numCC)[len(correlations)-numCC:len(correlations)]
            CC_for_Cycle.append(np.mean(correlations[top_indices]))

        CyclesCCs.append(CC_for_Cycle)
    
    end_time = time.time()
    
    ## obtain the common results
    # Quartiles
    Q1 = np.percentile(getTotalCC(CyclesCCs), 25)
    Q3 = np.percentile(getTotalCC(CyclesCCs), 75)
    
    # Interquartile Range
    IQR = Q3 - Q1
    
    # Bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Mean
    mean_value = np.mean(getTotalCC(CyclesCCs))
    
    # Display the results
    checkAndMakeFolder(path + "Arrays/")
    
    printandwrite(path + "results.txt", "Overall Results:\n", True, printIt=printIt)
    printandwrite(path + "results.txt", f"Time to calculate: {end_time - start_time}\n", printIt=True)
    printandwrite(path + "results.txt", f"Lower Bound (Outlier Threshold): {lower_bound}\n", printIt=printIt)
    printandwrite(path + "results.txt", f"First Quartile (Q1): {Q1}\n", printIt=printIt)
    printandwrite(path + "results.txt", f"Mean: {mean_value}\n", printIt=printIt)
    printandwrite(path + "results.txt", f"Third Quartile (Q3): {Q3}\n", printIt=printIt)
    printandwrite(path + "results.txt", f"Upper Bound (Outlier Threshold): {upper_bound}\n", printIt=printIt)
    printandwrite(path + "results.txt", f"Interquartile Range (IQR): {IQR}\n", printIt=printIt)
    
    # Save arrays.
    np.save(path + "Arrays/cycle_Results", np.array(CyclesCCs))
    np.save(path + "Arrays/sum_Results", np.array(getTotalCC(CyclesCCs)))

    return np.array(CyclesCCs), getTotalCC(CyclesCCs)


# In[ ]:

def percentage_in_range(values, low, high):
    # Count how many values fall within the range (inclusive)
    count_in_range = sum(1 for v in values if low <= v <= high)
    
    # Calculate the percentage
    percentage = (count_in_range / len(values)) * 100 if values.all() else 0
    
    return percentage

# In[ ]:

'''
Obtains the percentage of overlap for the real_CyclesCCs to the gen_CyclesCCs for each cycle and the overall for every cycle.
INPUT:
    real_CyclesCCs: Cycles average cross correlation when comparing real captured to real captured benign signals.
    gen_CyclesCCs: Cycles average cross correlation when comparing real captured to generated benign signals.
RETURN:
    Percent_Cycles: percent of overlap for each cycle.
    Percent_Overall: percent across the whole signal.
'''
def getPercentageFidelity(real_CyclesCCs, gen_CyclesCCs, path, printIt=False):
    percentage_Cycles = []
    for cycle in range(real_CyclesCCs.shape[0]):
        percentage_Cycles.append(percentage_in_range(gen_CyclesCCs[cycle], np.min(real_CyclesCCs[cycle]), np.max(real_CyclesCCs[cycle])))
            
    total_percentage = np.average(percentage_Cycles)
    
    printandwrite(path + "results.txt", "\n\nPercentage of Fidelity on Cycle Results:\n", printIt=printIt)
    for c in range(len(percentage_Cycles)):
        printandwrite(path + "results.txt", "Cycle-" + str(c) + ": " + str(percentage_Cycles[c]) + "\n", printIt=printIt)
    
    printandwrite(path + "results.txt", "\n\nPercentage of Fidelity Overall:\n", printIt=printIt)
    printandwrite(path + "results.txt", "Total: " + str(total_percentage) + "\n", printIt=printIt)
    
    return percentage_Cycles, total_percentage

# In[ ]:

'''
Obtains the peaks and an amount of amplitudes of x to the left and right of the peaks.

INPUT:
    dataset: dataset of signals.
    pad: area to check for peaks.
    amount_around: number of time indexs to get before and after the peaks.
    if_MajorOnly: boolean indicating on whether to get major peaks only. (I.E. each instruction contains two waves, the first contains a higher 
        amplitude that is the major peak, the second is lower and each second peak is usually around the same amplitude for every instruction.)
    clean_dataset: clean dataset to find the actual peak location. Required if necessary to be exact when noisy signals makes it impossible to find the correct peak location.
    use_Clean: boolean true or false to use clean.    
        
RETURNS:
    aroundPeak_Dataset: dataset of signals containing only the peaks and around the peak information.
'''
def getAroundPeaks(dataset, pad, amount_around, if_MajorOnly=False, clean_dataset=0, use_Clean=False, getLast=False, sameLoc=True):
    start_time = time.time()
    peaks = MapTool.getPeaksDataset(dataset, pad)
    
    amountPeaks = peaks.shape[1]
    #print(amountPeaks)
    
    end = 1
    if getLast:
        end = 2
    
    if use_Clean:
        checkDataset = np.copy(clean_dataset)
    else:
        checkDataset = np.copy(dataset)
    
    # check if first or second peak is a major peak.
    if peaks[0][0] > peaks[0][1]:
        start = 0
    else:
        start = 1
    
    aroundPeak_Dataset = []
    for s in range(dataset.shape[0]):
        signalAroundPeaks = [] 
        at = 0
        while start + at < amountPeaks-end:
            if sameLoc:
                loc = MapTool.getPeaksLoc(checkDataset[0], pad, start+at)
            else:
                loc = MapTool.getPeaksLoc(checkDataset[s], pad, start+at)
                
            signalAroundPeaks.extend(dataset[s][loc-amount_around:loc+amount_around+1])
            
            if len(dataset[s][loc-amount_around:loc+amount_around+1]) != 1+(amount_around*2):
                print("Took Wrong Peaks: ", len(dataset[s][loc-amount_around:loc+amount_around+1]))
                print("Loc: " + str(loc) + " at: ", str(at))
                print(dataset[s][loc-amount_around:loc+amount_around+1])
            at = at+1
            if if_MajorOnly:
                at = at+1
        #print(len(signalAroundPeaks))
        aroundPeak_Dataset.append(signalAroundPeaks)
        
    end_time = time.time()
    print("Time: ", end_time-start_time)
    
    return np.array(aroundPeak_Dataset)
        
# In[ ]

from scipy import signal as sciSignal

'''
Obtains the average of the euc per around peaks.
Assumes that the datasets will have the same number of instances. (Please check before hand.)
INPUT: 
    TS_Signals: (Time Series Signals) the EM signals of an execution path.
    Q_Signals: (Query Signals) the EM signals of an set instruction/s.
    Pad: Used for collecting the peaks. Is the average number of features between high and low peaks.
    numCC: the number of cross correlations to average. 

RETURNS:
    AroundPeaks_Eucs: list of lists of average Euclidean for each area around peaks.
    AroundPeaks_Euc_Total: lists of each signals avg euclean for each area around peaks added to each other.
'''
def EuclideanByPeaksAround(TS_Signals_Full, Q_Signals_Full, TS_Signals_Name, Q_Signals_Name, pad, numEucs, path, amountAround, if_MajorOnly=True, alreadyAroundPeakDataset=False, Graph=True, printIt=True, y_low=-2, y_high=2, sameLoc=True):
    start_time = time.time()
    
    checkAndMakeFolder(path + "Graphs/")
    
    # check if already around peak data and if not get it.
    if alreadyAroundPeakDataset == False:
        TS_Signals = getAroundPeaks(TS_Signals_Full, pad, amountAround, if_MajorOnly=if_MajorOnly, sameLoc=sameLoc)
        Q_Signals = getAroundPeaks(Q_Signals_Full, pad, amountAround, if_MajorOnly=if_MajorOnly, sameLoc=sameLoc)
        
        if Graph:
            Signal = TS_Signals_Full[0]
            time_Signal = np.arange(Signal.shape[0])
            plt.plot(time_Signal, Signal, label = TS_Signals_Name)
            Signal = Q_Signals_Full[0]
            time_Signal = np.arange(Signal.shape[0])
            plt.plot(time_Signal, Signal, label = Q_Signals_Name)
            plt.xlabel("Sample Index")
            plt.ylabel("Amplitude")
            plt.grid(axis='y')

            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
                      fancybox=True, shadow=True, ncol=6)
            plt.xlim([0,len(Signal)])
            plt.ylim(y_low,y_high)
            plt.savefig(path+ "Graphs/Compare_FullSignal.png", bbox_inches='tight')
            plt.show()
            plt.close()
            
            Signal = TS_Signals[0]
            time_Signal = np.arange(Signal.shape[0])
            plt.plot(time_Signal, Signal, label = TS_Signals_Name)
            Signal = Q_Signals[0]
            time_Signal = np.arange(Signal.shape[0])
            plt.plot(time_Signal, Signal, label = Q_Signals_Name)
            plt.xlabel("Sample Index")
            plt.ylabel("Amplitude")
            plt.grid(axis='y')

            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
                      fancybox=True, shadow=True, ncol=6)
            plt.xlim([0,len(Signal)])
            plt.ylim(0,y_high)
            plt.savefig(path+ "Graphs/Compare_AroundPeaksSignal.png", bbox_inches='tight')
            plt.show()
            plt.close()
    
    else:
        TS_Signals = np.copy(TS_Signals_Full)
        Q_Signals = np.copy(Q_Signals_Full)
        
        if Graph:
            Signal = TS_Signals[0]
            time_Signal = np.arange(Signal.shape[0])
            plt.plot(time_Signal, Signal, label = TS_Signals_Name)
            Signal = Q_Signals[0]
            time_Signal = np.arange(Signal.shape[0])
            plt.plot(time_Signal, Signal, label = Q_Signals_Name)
            plt.xlabel("Sample Index")
            plt.ylabel("Amplitude")
            plt.grid(axis='y')

            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
                      fancybox=True, shadow=True, ncol=6)
            plt.xlim([0,len(Signal)])
            plt.ylim(0,y_high)
            plt.savefig(path+ "Graphs/Compare_AroundPeaksSignal.png", bbox_inches='tight')
            plt.show()
            plt.close()
        
    if TS_Signals.shape[1] != Q_Signals.shape[1]:
        print("NOTICE: the number of amplitude instances are different. Will only check for the amount for the first given dataset, I.E. baseline.")
    
    # Obtain the cycles of each dataset
    AroundPeaks_Eucs = []
    AroundPeaks_Euc_Total = []

    for i in range(int(TS_Signals.shape[1]/(amountAround*2+1))):
        # Extract the i-th cycle for all TS and Q signals
        TS_signals_AroundPeaks = TS_Signals[:, i*amountAround:(i*amountAround)+amountAround+1]  # Shape: [num_TS]
        Q_signals_AroundPeaks = Q_Signals[:, i*amountAround:(i*amountAround)+amountAround+1]    # Shape: [num_Q]

        Euc_for_AroundPeak = []

        # Compute cross-correlations for each TS signal against all Q signals
        for Q_AroundPeak in Q_signals_AroundPeaks:
            # Vectorized correlation calculation using list comprehension
            eucs = np.array([Norm_EuclideanDistance(TS_AroundPeak, Q_AroundPeak) for TS_AroundPeak in TS_signals_AroundPeaks])

            # Efficient top-k selection
            top_indices = np.argpartition(eucs, numEucs)[len(eucs)-numEucs:len(eucs)]
            Euc_for_AroundPeak.append(np.mean(eucs[top_indices]))

        AroundPeaks_Euc_Total.append(Euc_for_AroundPeak)
    
    end_time = time.time()
    print("Total time: ", end_time-start_time)
    
    ## obtain the common results
    # Quartiles
    Q1 = np.percentile(getTotalCC(AroundPeaks_Euc_Total), 25)
    Q3 = np.percentile(getTotalCC(AroundPeaks_Euc_Total), 75)
    
    # Interquartile Range
    IQR = Q3 - Q1
    
    # Bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Mean
    mean_value = np.mean(getTotalCC(AroundPeaks_Euc_Total))
    
    # Display the results
    checkAndMakeFolder(path + "Arrays/")
    
    printandwrite(path + "results.txt", "Overall Results:\n", True, printIt=printIt)
    printandwrite(path + "results.txt", f"Time to calculate: {end_time - start_time}\n", printIt=printIt)
    printandwrite(path + "results.txt", f"Lower Bound (Outlier Threshold): {lower_bound}\n", printIt=printIt)
    printandwrite(path + "results.txt", f"First Quartile (Q1): {Q1}\n", printIt=printIt)
    printandwrite(path + "results.txt", f"Mean: {mean_value}\n", printIt=printIt)
    printandwrite(path + "results.txt", f"Third Quartile (Q3): {Q3}\n", printIt=printIt)
    printandwrite(path + "results.txt", f"Upper Bound (Outlier Threshold): {upper_bound}\n", printIt=printIt)
    printandwrite(path + "results.txt", f"Interquartile Range (IQR): {IQR}\n", printIt=printIt)
    
    # Save arrays.
    np.save(path + "Arrays/cycle_Results", np.array(AroundPeaks_Euc_Total))
    np.save(path + "Arrays/sum_Results", np.array(getTotalCC(AroundPeaks_Euc_Total)))

    return np.array(AroundPeaks_Euc_Total), getTotalCC(AroundPeaks_Euc_Total)

# In[ ]

def Norm_EuclideanDistance(X,Y):
    norm_X, norm_Y = normalize_lists(X, Y)
    return distance.euclidean(norm_X, norm_Y)

def normalize_lists(list1, list2):
    # Find the global min and max across both lists
    global_min = min(min(list1), min(list2))
    global_max = max(max(list1), max(list2))
    
    # Avoid division by zero in case all values are the same
    if global_max == global_min:
        return [0] * len(list1), [0] * len(list2)

    # Normalize each list
    normalize = lambda lst: [(x - global_min) / (global_max - global_min) for x in lst]
    
    return normalize(list1), normalize(list2)

#In [ ]

'''
Obtains the center location of the convex up.

INPUT:
    dataset: dataset of signals.
    pad: area to check for peaks.
    amount_around: number of time indexs to get before and after the peaks.
    if_MajorOnly: boolean indicating on whether to get major peaks only. (I.E. each instruction contains two waves, the first contains a higher 
        amplitude that is the major peak, the second is lower and each second peak is usually around the same amplitude for every instruction.)
    clean_dataset: clean dataset to find the actual peak location. Required if necessary to be exact when noisy signals makes it impossible to find the correct peak location.
    use_Clean: boolean true or false to use clean.    
        
RETURNS:
    aroundPeak_Dataset: dataset of signals containing only the peaks and around the peak information.
'''
def getCenters(dataset, pad, if_MajorOnly=False, clean_dataset=0, use_Clean=False):
    start_time = time.time()
    peaks = MapTool.getPeaksDataset(dataset, pad)
    
    amountPeaks = peaks.shape[1]
    #print(amountPeaks)
    
    end = 1
    if getLast:
        end = 2
    
    if use_Clean:
        checkDataset = np.copy(clean_dataset)
    else:
        checkDataset = np.copy(dataset)
    
    # check if first or second peak is a major peak.
    if peaks[0][0] > peaks[0][1]:
        start = 0
    else:
        start = 1
    
    aroundPeak_Dataset = []
    for s in range(dataset.shape[0]):
        signalAroundPeaks = [] 
        at = 0
        while start + at < amountPeaks-end:
            if sameLoc:
                loc = MapTool.getPeaksLoc(checkDataset[0], pad, start+at)
            else:
                loc = MapTool.getPeaksLoc(checkDataset[s], pad, start+at)
                
            signalAroundPeaks.extend(dataset[s][loc-amount_around:loc+amount_around+1])
            
            if len(dataset[s][loc-amount_around:loc+amount_around+1]) != 1+(amount_around*2):
                print("Took Wrong Peaks: ", len(dataset[s][loc-amount_around:loc+amount_around+1]))
                print("Loc: " + str(loc) + " at: ", str(at))
                print(dataset[s][loc-amount_around:loc+amount_around+1])
            at = at+1
            if if_MajorOnly:
                at = at+1
        #print(len(signalAroundPeaks))
        aroundPeak_Dataset.append(signalAroundPeaks)
        
    end_time = time.time()
    print("Time: ", end_time-start_time)
    
    return np.array(aroundPeak_Dataset)
        
    
def getCenterConvUp(signal,checkloc, midpoint, buffer_lim=1):
    # Identify len of the convex
    check=True
    while check:
        if checkloc < len(signal)-1:
            if signal[checkloc] < midpoint:
                checkloc = checkloc+1
            else:
                check=False
        else:
            check=False
            
    startloc = checkloc
    endloc = checkloc
    end = False
        
    while endloc+1 < len(signal) and end==False:
        if signal[endloc+1] > midpoint:
            endloc = endloc+1
        else:
            if endloc+2 < len(signal):
                if signal[endloc+2] > midpoint:
                    endloc=endloc+1
                else:
                    end = True
            else:
                end=True
    #Identify the centerbased on the startloc and endloc
    half = (endloc-startloc)//2
    return startloc+half, endloc

def getCenterConvDown(signal,checkloc, midpoint):
    # Identify len of the convex    
    check=True
    while check:
        if signal[checkloc] > midpoint:
            checkloc = checkloc+1
        else:
            check=False
            
    startloc = checkloc
    endloc = checkloc
    end = False
            
    while endloc+1 < len(signal) and end==False:
        if signal[endloc+1] < midpoint:
            endloc = endloc+1
        else:
            if endloc+2 < len(signal): 
                if signal[endloc+2] < midpoint:
                    endloc=endloc+1
                else:
                    end = True
            else:
                end = True
    #Identify the centerbased on the startloc and endloc
    half = (endloc-startloc)//2
    return startloc+half, endloc

'''
Obtains the center location of the convex up.

INPUT:
    signal: dataset of signals.
    pad: area to check for peaks.
    if_MajorOnly: boolean indicating on whether to get major peaks only. (I.E. each instruction contains two waves, the first contains a higher 
        amplitude that is the major peak, the second is lower and each second peak is usually around the same amplitude for every instruction.)
    clean_dataset: clean dataset to find the actual peak location. Required if necessary to be exact when noisy signals makes it impossible to find the correct peak location.
    use_Clean: boolean true or false to use clean.    
        
RETURNS:
    aroundPeak_Dataset: dataset of signals containing only the peaks and around the peak information.
'''
def getCenterLocs(signal, pad, if_MajorOnly=True, midpoint=0, mini_len=5):
    start_time = time.time()
    
    center_Locs = []
    checkloc = 0
    start =True
    Up=True
    majorPeak=True
    while checkloc+1 < len(signal):
        if Up:
            oldcheck = checkloc
            center, checkloc = getCenterConvUp(signal,checkloc, midpoint)
            #print("Center: ", center)
            #print("CheckLoc: ", checkloc)
            if majorPeak:
                if checkloc-oldcheck >= mini_len:
                    center_Locs.append(center)
                    majorPeak=False
            else:
                if if_MajorOnly==False:
                    if checkloc-oldcheck >= mini_len:
                        center_Locs.append(center)
                if checkloc-oldcheck > mini_len:
                    majorPeak=True
            Up=False
        else:
            center, checkloc = getCenterConvDown(signal,checkloc, midpoint)
            #print("Center: ", center)
            #print("CheckLoc: ", checkloc)
            Up=True
    
    return center_Locs
        
    
def obtainSignalCenters(signal, centerLocs, amount):
    new_Signal = []
    for centerLoc in centerLocs:
        #print(centerLoc)
        new_Signal.extend(signal[centerLoc-amount:centerLoc+amount+1])
    return new_Signal

def obtainSignalCentersDataset(signals, pad, amount, if_MajorOnly=True, midpoint=0, mini_len=5):
    new_Signals = []
    centerLocs_Set = getCenterLocsDataset(signals, pad, if_MajorOnly, midpoint, mini_len)
    
    for i in range(signals.shape[0]):
        new_Signals.append(obtainSignalCenters(signals[i], centerLocs_Set[i], amount))
    
    return np.array(new_Signals)
    

'''
Obtains the center location of the convex up across the entire dataset.

INPUT:
    signal: dataset of signals.
    pad: area to check for peaks.
    if_MajorOnly: boolean indicating on whether to get major peaks only. (I.E. each instruction contains two waves, the first contains a higher 
        amplitude that is the major peak, the second is lower and each second peak is usually around the same amplitude for every instruction.)
    clean_dataset: clean dataset to find the actual peak location. Required if necessary to be exact when noisy signals makes it impossible to find the correct peak location.
    use_Clean: boolean true or false to use clean.    
        
RETURNS:
    aroundPeak_Dataset: dataset of signals containing only the peaks and around the peak information.
'''
def getCenterLocsDataset(dataset, pad, if_MajorOnly=True, midpoint=0, mini_len=5):
    start_time = time.time()
    
    center_Locs_Dataset = []
    for signal in dataset:
        center_Locs_Dataset.append(getCenterLocs(signal, pad, if_MajorOnly, midpoint, mini_len))
        #print(len(getCenterLocs(signal, pad, if_MajorOnly=False, midpoint=0)))
    
    end_time = time.time()
    print("Time: ", end_time - start_time)
    
    return np.array(center_Locs_Dataset)

def extractTheCenterAround_Ith(dataset, centerLocs, centerLocs_i, amount):
    aroundCenters = []
    for s in range(dataset.shape[0]):
        loc = centerLocs[s][centerLocs_i]
        aroundCenter = dataset[s][loc-amount:loc+amount+1]
        aroundCenters.append(aroundCenter)
    return np.array(aroundCenters)

from scipy import signal as sciSignal

'''
Obtains the average of the euc per around center locations.
Assumes that the datasets will have the same number of instances. (Please check before hand.)
INPUT: 
    TS_Signals: (Time Series Signals) the EM signals of an execution path.
    Q_Signals: (Query Signals) the EM signals of an set instruction/s.
    Pad: Used for collecting the peaks. Is the average number of features between high and low peaks.
    numCC: the number of euclidean to average. 

RETURNS:
    AroundCenters_Eucs: list of lists of average Euclidean for each area around peaks.
    AroundCenters_Euc_Total: lists of each signals avg euclean for each area around peaks added to each other.
'''
def EuclideanByCenter(TS_Signals, Q_Signals, TS_Signals_Name, Q_Signals_Name, pad, numEucs, path, amountAround, if_MajorOnly=True, Graph=True, printIt=True, y_low=-2, y_high=2, mini_len=5, midpoint=0):
    start_time = time.time()
    
    checkAndMakeFolder(path + "Graphs/")
    
    # check if already around peak data and if not get it.
    TS_Signals_centerLocs = getCenterLocsDataset(TS_Signals, pad, if_MajorOnly, midpoint, mini_len)
    Q_Signals_centerLocs = getCenterLocsDataset(Q_Signals, pad, if_MajorOnly, midpoint, mini_len)
    
    if Graph:
            Signal = TS_Signals[0]
            time_Signal = np.arange(Signal.shape[0])
            plt.plot(time_Signal, Signal, label = TS_Signals_Name)
            Signal = Q_Signals[0]
            time_Signal = np.arange(Signal.shape[0])
            plt.plot(time_Signal, Signal, label = Q_Signals_Name)
            plt.xlabel("Sample Index")
            plt.ylabel("Amplitude")
            plt.grid(axis='y')

            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
                      fancybox=True, shadow=True, ncol=6)
            plt.xlim([0,len(Signal)])
            plt.ylim(y_low,y_high)
            plt.savefig(path+ "Graphs/Compare_FullSignal.png", bbox_inches='tight')
            plt.show()
            plt.close()
            
            signal_1 = obtainSignalCenters(TS_Signals[0], TS_Signals_centerLocs[0], amountAround)
            signal_2 = obtainSignalCenters(Q_Signals[0], Q_Signals_centerLocs[0], amountAround)
            
            Signal = signal_1
            time_Signal = np.arange(len(Signal))
            plt.plot(time_Signal, Signal, label = TS_Signals_Name)
            Signal = signal_2
            time_Signal = np.arange(len(Signal))
            plt.plot(time_Signal, Signal, label = Q_Signals_Name)
            plt.xlabel("Sample Index")
            plt.ylabel("Amplitude")
            plt.grid(axis='y')

            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
                      fancybox=True, shadow=True, ncol=6)
            plt.xlim([0,len(Signal)])
            plt.ylim(0,y_high)
            plt.savefig(path+ "Graphs/Compare_AroundCentersSignal.png", bbox_inches='tight')
            plt.show()
            plt.close()
        
    # Obtain the cycles of each dataset
    AroundCenters_Eucs = []
    AroundCenters_Euc_Total = []

    for i in range(int(TS_Signals_centerLocs.shape[1])):
        # Extract the i-th cycle for all TS and Q signals
        TS_Signals_AroundCenters = extractTheCenterAround_Ith(TS_Signals, TS_Signals_centerLocs, i, amountAround)  # Shape: [num_TS]
        Q_Signals_AroundCenters = extractTheCenterAround_Ith(Q_Signals, Q_Signals_centerLocs, i, amountAround)    # Shape: [num_Q]

        Euc_for_AroundCenter = []

        # Compute cross-correlations for each TS signal against all Q signals
        for Q_AroundCenter in Q_Signals_AroundCenters:
            # Vectorized correlation calculation using list comprehension
            #eucs = np.array([Norm_EuclideanDistance(TS_AroundCenter, Q_AroundCenter) for TS_AroundCenter in TS_signals_AroundCenters])
            eucs = np.array([distance.euclidean(TS_AroundCenter, Q_AroundCenter) for TS_AroundCenter in TS_Signals_AroundCenters])

            # Efficient top-k selection
            top_indices = np.argpartition(eucs, numEucs)[len(eucs)-numEucs:len(eucs)]
            Euc_for_AroundCenter.append(np.mean(eucs[top_indices]))

        AroundCenters_Euc_Total.append(Euc_for_AroundCenter)
    
    end_time = time.time()
    print("Total time: ", end_time-start_time)
    
    ## obtain the common results
    # Quartiles
    Q1 = np.percentile(getTotalCC(AroundCenters_Euc_Total), 25)
    Q3 = np.percentile(getTotalCC(AroundCenters_Euc_Total), 75)
    
    # Interquartile Range
    IQR = Q3 - Q1
    
    # Bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Mean
    mean_value = np.mean(getTotalCC(AroundCenters_Euc_Total))
    
    # Display the results
    checkAndMakeFolder(path + "Arrays/")
    
    printandwrite(path + "results.txt", "Overall Results:\n", True, printIt=printIt)
    printandwrite(path + "results.txt", f"Time to calculate: {end_time - start_time}\n", printIt=printIt)
    printandwrite(path + "results.txt", f"Lower Bound (Outlier Threshold): {lower_bound}\n", printIt=printIt)
    printandwrite(path + "results.txt", f"First Quartile (Q1): {Q1}\n", printIt=printIt)
    printandwrite(path + "results.txt", f"Mean: {mean_value}\n", printIt=printIt)
    printandwrite(path + "results.txt", f"Third Quartile (Q3): {Q3}\n", printIt=printIt)
    printandwrite(path + "results.txt", f"Upper Bound (Outlier Threshold): {upper_bound}\n", printIt=printIt)
    printandwrite(path + "results.txt", f"Interquartile Range (IQR): {IQR}\n", printIt=printIt)
    
    # Save arrays.
    np.save(path + "Arrays/cycle_Results", np.array(AroundCenters_Euc_Total))
    np.save(path + "Arrays/sum_Results", np.array(getTotalCC(AroundCenters_Euc_Total)))

    return np.array(AroundCenters_Euc_Total), getTotalCC(AroundCenters_Euc_Total)