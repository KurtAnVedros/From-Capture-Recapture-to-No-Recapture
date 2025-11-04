#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import random
import tensorflow as tf
import numpy as np
import timeit
import scipy.io as sio
import csv
import os
import math 
import random as rand
from matplotlib import pyplot as plt

from scipy.spatial import distance
from scipy.spatial.distance import euclidean
#from fastdtw import fastdtw

from Modules.Tools import MapTool


# In[ ]:


params = {'legend.fontsize': 50,
          'figure.figsize': (45, 15),
         'axes.labelsize': 60,
         'axes.titlesize':70,
         'xtick.labelsize':50,
         'ytick.labelsize':50,
         'lines.linewidth': 5}
plt.rcParams.update(params)


# In[ ]:


# Gather all information of the peak in all signals.
def gatherInfoPeaks(Signals, peakLoc, pad):
    dataOfPeaks = []
    for signal in Signals:
        dataOfPeaks.append(signal[MapTool.getPeaksLoc(signal, pad, peakLoc)])
    dataOfPeaks = np.array((dataOfPeaks))

    return dataOfPeaks


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


def boxPlotData(dictionary, xName, yName, title, y_limit_low, y_limit_high,
                linewidth =10, markersize = 20, figsize= (30, 30), width = 0.6, folder= ""):
    # DictionaryFull should contain: [Signals, peakLoc, name, dataOfPeaks]
    
    # Collect the data instance.
    data = []
    for info in dictionary:
        data.append(info[3])
    
    # Basic figure settings
    fig = plt.figure(figsize = figsize)   
    ax = fig.add_axes([0, 0, 1, 1])
    
    # Box plot settings
    flierprops = dict(markersize=markersize, markerfacecolor='yellow')
    boxprops = dict(linewidth = linewidth, c = 'g')
    whiskerprops = dict(linewidth = linewidth, c = 'b')
    capprops = dict(linewidth = linewidth)
    medianprops = dict(linewidth = linewidth)
    meanprops = dict(linewidth = linewidth)
    
    # Create a Box Plot of the data.
    bp = ax.boxplot(data, widths = width, meanprops = meanprops, capprops = capprops, boxprops=boxprops
                , medianprops = medianprops, whiskerprops=whiskerprops, flierprops=flierprops)
    
    if dictionary.shape[0] > 1:
        names =[]
        for info in dictionary:
            names.append(info[2])
        ax.set_xticklabels(names)
    
    for label in ax.xaxis.get_ticklabels():
        label.set_rotation(-30)
        
    for line in ax.xaxis.get_ticklines():
        line.set_markersize(15)
    
    #ax.set_title('axes title')
    ax.set_xlabel(xName)
    ax.set_ylabel(yName)
    
    plt.title(title,fontweight="bold")
    
    plt.ylim(y_limit_low, y_limit_high)
    
    plt.grid(axis ='y')
    
    if folder == "":
        plt.show()
    else:
        plt.savefig(folder, bbox_inches='tight')


# In[ ]:


def boxPlotDataSeparate(dictionary, xName, yName, title, y_limit_low, y_limit_high,
                linewidth =10, markersize = 20, figsize= (30, 30), width = 0.6, folder= "", printLengths=False):
    # DictionaryFull should contain: [Data, name]
    
    # Collect the data instance.
    data = []
    for info in dictionary:
        temp = info[0].flatten()
        data.append(temp)
    
    if printLengths:
        for i in range(len(data)):
            print(data[i].shape)
    
    
    # Basic figure settings
    fig = plt.figure(figsize = figsize)   
    ax = fig.add_axes([0, 0, 1, 1])
    
    # Box plot settings
    flierprops = dict(markersize=markersize, markerfacecolor='yellow')
    boxprops = dict(linewidth = linewidth, c = 'g')
    whiskerprops = dict(linewidth = linewidth, c = 'b')
    capprops = dict(linewidth = linewidth)
    medianprops = dict(linewidth = linewidth)
    meanprops = dict(linewidth = linewidth)
    
    # Create a Box Plot of the data.
    bp = ax.boxplot(data, widths = width, meanprops = meanprops, capprops = capprops, boxprops=boxprops
                , medianprops = medianprops, whiskerprops=whiskerprops, flierprops=flierprops)
    
    names =[]
    for info in dictionary:
        names.append(info[1])
    ax.set_xticklabels(names)
    
    #for label in ax.xaxis.get_ticklabels():
    #    label.set_rotation(-30)
        
    for line in ax.xaxis.get_ticklines():
        line.set_markersize(15)
    
    #ax.set_title('axes title')
    ax.set_xlabel(xName,fontweight="bold")
    ax.set_ylabel(yName,fontweight="bold")
    
    plt.title(title,fontweight="bold")
    
    plt.ylim(y_limit_low, y_limit_high)
    
    plt.grid(axis ='y')
    
    if folder == "":
        plt.show()
    else:
        plt.savefig(folder, bbox_inches='tight')


# In[ ]:


def boxPlotInstPeaks(dictionary, xName, yName, TS_Peak_name,  Q_Peak_name, end_title, y_limit_low=0, y_limit_high=0,
                linewidth =10, markersize = 20, figsize= (30, 30), width = 0.6, folder = ""):
    # DictionaryFull should contain: [Signals, peakLoc, name, dataOfPeaks]
    
    title = "ED Comparison: " + TS_Peak_name + " and " + Q_Peak_name + end_title
    
    # Collect the data instance.
    data = []
    for info in dictionary:
        data.append(info[2])
    
    # Basic figure settings
    fig = plt.figure(figsize = figsize)   
    ax = fig.add_axes([0, 0, 1, 1])
    
    # Box plot settings
    flierprops = dict(markersize=markersize, markerfacecolor='yellow')
    boxprops = dict(linewidth = linewidth, c = 'g')
    whiskerprops = dict(linewidth = linewidth, c = 'b')
    capprops = dict(linewidth = linewidth)
    medianprops = dict(linewidth = linewidth)
    meanprops = dict(linewidth = linewidth)
    
    # Create a Box Plot of the data.
    bp = ax.boxplot(data, widths = width, meanprops = meanprops, capprops = capprops, boxprops=boxprops
                , medianprops = medianprops, whiskerprops=whiskerprops, flierprops=flierprops)
    
    if dictionary.shape[0] > 1:
        names =[]
        for info in dictionary:
            names.append(info[1])
        ax.set_xticklabels(names)
    
    #for label in ax.xaxis.get_ticklabels():
    #    label.set_rotation(-30)
        
    for line in ax.xaxis.get_ticklines():
        line.set_markersize(15)
    
    #ax.set_title('axes title')
    ax.set_xlabel(xName,fontweight="bold")
    ax.set_ylabel(yName,fontweight="bold")
    
    plt.title(title,fontweight="bold")
    
    if y_limit_high != 0:
        plt.ylim(y_limit_low, y_limit_high)
    
    plt.grid(axis ='y')
    
    if folder == "":
        plt.show()
    else:
        fileName = folder + "BPlotDist_" + TS_Peak_name + "_vs_" + Q_Peak_name + ".png"
        plt.savefig(fileName, bbox_inches='tight')
        plt.show()


# In[ ]:


# Creates a box plot of the Signals at a peak location.
# does this for multiple
def createBoxPeakLoc(dictionary, xName, yName, title, y_limit_low, y_limit_high):
    # Dictionary should contain: [Signals, peakLoc, name, pad]
    # DictionaryFull should contain: [Signals, peakLoc, name, dataOfPeaks]
    dictionaryFull = []
    
    # Gathers all the information of the peak
    for i in range(dictionary.shape[0]):
        dataOfPeaks = gatherInfoPeaks(dictionary[i][0], dictionary[i][1], dictionary[i][3])
        # Fill dictionaryFull Info
        Signals = dictionary[i][0]
        peakLoc = dictionary[i][1]
        name =  dictionary[i][2]
        data = dataOfPeaks
        dictionaryRow = np.array((Signals, peakLoc, name, data), dtype=object)
        dictionaryFull.append(dictionaryRow)
    
    dictionaryFull = np.array((dictionaryFull))
    
    boxPlotData(dictionaryFull, xName, yName, title, y_limit_low, y_limit_high)


# In[ ]:


'''
Create two box plots, one of the query instructions and one of the time-series signals instructions
'''
'''
def compareBoxPlot(Q_dictionary, Q_title, TS_dictionary, TS_title, xName, yName, simRange, y_limit_low, y_limit_high, pad):
    # Dictionary should contain: [Signals, peakLoc, name, pad]
    # DictionaryFull should contain: [Signals, peakLoc, name, dataOfPeaks]
    Q_dictionaryFull = []
    
    # Gathers all the information of the peak
    for i in range(Q_dictionary.shape[0]):
        dataOfPeaks = gatherInfoPeaks(Q_dictionary[i][0], Q_dictionary[i][1], Q_dictionary[i][3])
        # Fill dictionaryFull Info
        Signals = Q_dictionary[i][0]
        peakLoc = Q_dictionary[i][1]
        name =  Q_dictionary[i][2]
        data = dataOfPeaks
        dictionaryRow = np.array((Signals, peakLoc, name, data), dtype=object)
        Q_dictionaryFull.append(dictionaryRow)
    
    Q_dictionaryFull = np.array((Q_dictionaryFull))
    
    # Dictionary should contain: [Signals, peakLoc, name, pad]
    # DictionaryFull should contain: [Signals, peakLoc, name, dataOfPeaks]
    TS_dictionaryFull = []
    
    # Gathers all the information of the peak
    for i in range(TS_dictionary.shape[0]):
        dataOfPeaks = gatherInfoPeaks(TS_dictionary[i][0], TS_dictionary[i][1], TS_dictionary[i][3])
        # Fill dictionaryFull Info
        Signals = TS_dictionary[i][0]
        peakLoc = TS_dictionary[i][1]
        name =  TS_dictionary[i][2]
        data = dataOfPeaks
        dictionaryRow = np.array((Signals, peakLoc, name, data), dtype=object)
        TS_dictionaryFull.append(dictionaryRow)
    
    TS_dictionaryFull = np.array((TS_dictionaryFull))
    
    boxPlotData(Q_dictionaryFull, xName, yName, Q_title, y_limit_low, y_limit_high)
    boxPlotData(TS_dictionaryFull, xName, yName, TS_title, y_limit_low, y_limit_high)
    
    print("Analyzing similar amplidue peak ranges")
    print("Queries peak ranges")
    Q_Medians = []
    for i in range(Q_dictionaryFull.shape[0]):
        median = getBoxInfo(Q_dictionaryFull[i][0], Q_dictionaryFull[i][1], Q_dictionaryFull[i][2], pad)
        Q_Medians.append(median)
    Q_Medians = np.array(Q_Medians)
    
    print("")
    print("Time-Series peak ranges")
    TS_Medians = []
    for i in range(TS_dictionaryFull.shape[0]):
        median = getBoxInfo(TS_dictionaryFull[i][0], TS_dictionaryFull[i][1], TS_dictionaryFull[i][2], pad)
        TS_Medians.append(median)
    TS_Medians = np.array(TS_Medians)
    
    length = TS_Median.shape[0] - Q_Median.shape[0]
    similarRange = []
    for l in range(length):
        true = 1
        for q in range(Q_Median.shape[0]):
            if TS_Medians[l+q] > Q_Medians[q] + simRange and TS_Medians[l+q] < Q_Medians[q] - simRange:
                true = 0
        if true == 1:
            similarRange.append(l)
    
    similarRange = np.array(similarRange)
    if similarRange.shape[0] == 0:
        print("No similar sequence of instructions found.")
    
    print("Found " + similarRange.shape[0] + " similar sequence of instructions")
    print("Query with instruction medians: ")
    for i in range(Q_dictionaryFull.shape[0]):
        median = getBoxInfo(Q_dictionaryFull[i][0], Q_dictionaryFull[i][1], Q_dictionaryFull[i][2], pad)

    print("")
    print("Is similar to the following sequences of instuctions in the time-series signals.")
    count= 0
    for s in range(similarRange):
        count= count+1
        print(str(count) + " similar instructions."
        for i in range(Q_dictionaryFull.shape[0]):
            median = getBoxInfo(TS_dictionaryFull[s+i][0], TS_dictionaryFull[s+i][1], TS_dictionaryFull[s+i][2], pad)
    
    print("")
    print("BoxPlot comparison")
    dictionaryFull = []
    
    # Gathers all the information of the peak
    for i in range(Q_dictionary.shape[0]):
        dataOfPeaks = gatherInfoPeaks(Q_dictionary[i][0], Q_dictionary[i][1], Q_dictionary[i][3])
        # Fill dictionaryFull Info
        Signals = Q_dictionary[i][0]
        peakLoc = Q_dictionary[i][1]
        name =  Q_dictionary[i][2]
        data = dataOfPeaks
        dictionaryRow = np.array((Signals, peakLoc, name, data), dtype=object)
        dictionaryFull.append(dictionaryRow)
    
    for s in range(similarRange):
        count= count+1
        print(str(count) + " similar instructions."
        #for i in range(Q_dictionaryFull.shape[0]):
    
    
    Q_dictionaryFull = np.array((Q_dictionaryFull))
'''


# In[ ]:


# Creates a box plot of the Signals at a peak location.
# does this for multiple
def boxPlotComp(dictionary, xName, yName, title, y_limit_low, y_limit_high, pad, folder = "", toFileSave = 0):
    # Dictionary should contain: [Signals, peakLoc, name, pad]
    # DictionaryFull should contain: [Signals, peakLoc, name, dataOfPeaks]
    dictionaryFull = []
    
    # Gathers all the information of the peak
    for i in range(dictionary.shape[0]):
        dataOfPeaks = gatherInfoPeaks(dictionary[i][0], dictionary[i][1], dictionary[i][3])
        # Fill dictionaryFull Info
        Signals = dictionary[i][0]
        peakLoc = dictionary[i][1]
        name =  dictionary[i][2]
        data = dataOfPeaks
        dictionaryRow = np.array((Signals, peakLoc, name, data), dtype=object)
        dictionaryFull.append(dictionaryRow)
    
    dictionaryFull = np.array((dictionaryFull))
    
    folder_BP = folder + "_BPlot.png"
    
    boxPlotData(dictionaryFull, xName, yName, title, y_limit_low, y_limit_high, folder = folder_BP)
    
    
    
    info = []
    for i in range(dictionaryFull.shape[0]):
        median = getBoxInfo(dictionaryFull[i][0], dictionaryFull[i][1], dictionaryFull[i][2], pad)
        name = dictionary[i][2]
        infoRow = np.array((median, name), dtype=object)
        info.append(infoRow)
    info = np.array((info))
    
    info_s = info[info[:, 0].argsort()]
    
    #print(info_s)
    #print(info_s.shape)
    
    folder_Info = folder + "_Info.txt"
    
    printOrWriteInfo(folder_Info, "Sorted Median\n", toFileSave, firstLine = True)
    for n in range(info_s.shape[0]):
        string = info_s[n][1] + ": " + str(info_s[n][0]) +"\n"
        printOrWriteInfo(folder_Info, string , toFileSave, firstLine = False)


# In[ ]:


# Creates a box plot of the Signals at a peak location.
# does this for multiple
def boxPlotCompTwoDatasets(Q_dictionary, Q_title, TS_dictionary, TS_title, xName, yName,
                           y_limit_low, y_limit_high, pad):
    # Dictionary should contain: [Signals, peakLoc, name, pad]
    # DictionaryFull should contain: [Signals, peakLoc, name, dataOfPeaks]
    Q_dictionaryFull = []
    
    # Gathers all the information of the peak
    for i in range(Q_dictionary.shape[0]):
        dataOfPeaks = gatherInfoPeaks(Q_dictionary[i][0], Q_dictionary[i][1], Q_dictionary[i][3])
        # Fill dictionaryFull Info
        Signals = Q_dictionary[i][0]
        peakLoc = Q_dictionary[i][1]
        name =  Q_dictionary[i][2]
        data = dataOfPeaks
        Q_dictionaryRow = np.array((Signals, peakLoc, name, data), dtype=object)
        Q_dictionaryFull.append(Q_dictionaryRow)
    
    Q_dictionaryFull = np.array((Q_dictionaryFull))
    
    boxPlotData(Q_dictionaryFull, xName, yName, Q_title, y_limit_low, y_limit_high)
    
    # Dictionary should contain: [Signals, peakLoc, name, pad]
    # DictionaryFull should contain: [Signals, peakLoc, name, dataOfPeaks]
    TS_dictionaryFull = []
    
    # Gathers all the information of the peak
    for i in range(TS_dictionary.shape[0]):
        dataOfPeaks = gatherInfoPeaks(TS_dictionary[i][0], TS_dictionary[i][1], TS_dictionary[i][3])
        # Fill dictionaryFull Info
        Signals = TS_dictionary[i][0]
        peakLoc = TS_dictionary[i][1]
        name =  TS_dictionary[i][2]
        data = dataOfPeaks
        TS_dictionaryRow = np.array((Signals, peakLoc, name, data), dtype=object)
        TS_dictionaryFull.append(TS_dictionaryRow)
    
    TS_dictionaryFull = np.array((TS_dictionaryFull))
    
    boxPlotData(TS_dictionaryFull, xName, yName, TS_title, y_limit_low, y_limit_high)
    
    info = []
    for i in range(Q_dictionaryFull.shape[0]):
        median = getBoxInfo(Q_dictionaryFull[i][0], Q_dictionaryFull[i][1], Q_dictionaryFull[i][2], pad)
        name = Q_dictionary[i][2]
        infoRow = np.array((median, name), dtype=object)
        info.append(infoRow)
        
    for i in range(TS_dictionaryFull.shape[0]):
        median = getBoxInfo(TS_dictionaryFull[i][0], TS_dictionaryFull[i][1], TS_dictionaryFull[i][2], pad)
        name = TS_dictionary[i][2]
        infoRow = np.array((median, name), dtype=object)
        info.append(infoRow)
        
    info = np.array((info))
    
    info_s = info[info[:, 0].argsort()]
    
    #print(info_s)
    #print(info_s.shape)
    
    print("Sorted Median")
    for n in range(info_s.shape[0]):
        print(info_s[n][1] + ": " + str(info_s[n][0]))


# In[ ]:


# Creates a box plot of the Signals at a peak location.
# does this for multiple
def boxPlotCompThreeDatasets(A_dictionary, A_title, B_dictionary, B_title, C_dictionary, C_title, xName, yName,
                           y_limit_low, y_limit_high, pad):
    # Dictionary should contain: [Signals, peakLoc, name, pad]
    # DictionaryFull should contain: [Signals, peakLoc, name, dataOfPeaks]
    A_dictionaryFull = []
    
    # Gathers all the information of the peak
    for i in range(A_dictionary.shape[0]):
        dataOfPeaks = gatherInfoPeaks(A_dictionary[i][0], A_dictionary[i][1], A_dictionary[i][3])
        # Fill dictionaryFull Info
        Signals = A_dictionary[i][0]
        peakLoc = A_dictionary[i][1]
        name =  A_dictionary[i][2]
        data = dataOfPeaks
        A_dictionaryRow = np.array((Signals, peakLoc, name, data), dtype=object)
        A_dictionaryFull.append(A_dictionaryRow)
    
    A_dictionaryFull = np.array((A_dictionaryFull))
    
    boxPlotData(A_dictionaryFull, xName, yName, A_title, y_limit_low, y_limit_high)
    
    # Dictionary should contain: [Signals, peakLoc, name, pad]
    # DictionaryFull should contain: [Signals, peakLoc, name, dataOfPeaks]
    B_dictionaryFull = []
    
    # Gathers all the information of the peak
    for i in range(B_dictionary.shape[0]):
        dataOfPeaks = gatherInfoPeaks(B_dictionary[i][0], B_dictionary[i][1], B_dictionary[i][3])
        # Fill dictionaryFull Info
        Signals = B_dictionary[i][0]
        peakLoc = B_dictionary[i][1]
        name =  B_dictionary[i][2]
        data = dataOfPeaks
        B_dictionaryRow = np.array((Signals, peakLoc, name, data), dtype=object)
        B_dictionaryFull.append(B_dictionaryRow)
    
    B_dictionaryFull = np.array((B_dictionaryFull))
    
    boxPlotData(B_dictionaryFull, xName, yName, B_title, y_limit_low, y_limit_high)
    
    # Dictionary should contain: [Signals, peakLoc, name, pad]
    # DictionaryFull should contain: [Signals, peakLoc, name, dataOfPeaks]
    C_dictionaryFull = []
    
    # Gathers all the information of the peak
    for i in range(C_dictionary.shape[0]):
        dataOfPeaks = gatherInfoPeaks(C_dictionary[i][0], C_dictionary[i][1], C_dictionary[i][3])
        # Fill dictionaryFull Info
        Signals = C_dictionary[i][0]
        peakLoc = C_dictionary[i][1]
        name =  C_dictionary[i][2]
        data = dataOfPeaks
        C_dictionaryRow = np.array((Signals, peakLoc, name, data), dtype=object)
        C_dictionaryFull.append(C_dictionaryRow)
    
    C_dictionaryFull = np.array((C_dictionaryFull))
    
    boxPlotData(C_dictionaryFull, xName, yName, C_title, y_limit_low, y_limit_high)
    
    info = []
    for i in range(A_dictionaryFull.shape[0]):
        median = getBoxInfo(A_dictionaryFull[i][0], A_dictionaryFull[i][1], A_dictionaryFull[i][2], pad)
        name = A_dictionary[i][2]
        infoRow = np.array((median, name), dtype=object)
        info.append(infoRow)
        
    for i in range(B_dictionaryFull.shape[0]):
        median = getBoxInfo(B_dictionaryFull[i][0], B_dictionaryFull[i][1], B_dictionaryFull[i][2], pad)
        name = B_dictionary[i][2]
        infoRow = np.array((median, name), dtype=object)
        info.append(infoRow)
    
    for i in range(C_dictionaryFull.shape[0]):
        median = getBoxInfo(C_dictionaryFull[i][0], C_dictionaryFull[i][1], C_dictionaryFull[i][2], pad)
        name = C_dictionary[i][2]
        infoRow = np.array((median, name), dtype=object)
        info.append(infoRow)
        
    info = np.array((info))
    
    info_s = info[info[:, 0].argsort()]
    
    #print(info_s)
    #print(info_s.shape)
    
    print("Sorted Median")
    for n in range(info_s.shape[0]):
        print(info_s[n][1] + ": " + str(info_s[n][0]))


# In[ ]:


def getBoxInfo(Signals, peakLoc, Name, pad):
    data = gatherInfoPeaks(Signals, peakLoc, pad)
    q1, median, q3 = np.percentile(data, [25, 50, 75])
    iqr = 1.5*(q3-q1)
    if data.min() > q1-iqr:
        minimum = data.min()
    else:
        minimum = q1-iqr
    if data.max() < q3 + iqr:
        maximum = data.max()
    else:
        maximum = q3+iqr
        
    #print(Name + " INFO: " + "Min: " + str(minimum) + " Q1: " + str(q1) + " Median: " + 
    #      str(median) + " Q3: " + str(q3) + " Max: " + str(maximum))
    #print(Name + " INFO: " + " Median: " + str(median))
    return median


# In[ ]:




