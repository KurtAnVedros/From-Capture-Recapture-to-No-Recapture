#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import random
import tensorflow as tf
import numpy as np
import timeit
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

from scipy.signal.signaltools import wiener
from sklearn.neighbors import KNeighborsRegressor
import warnings
from scipy.signal import savgol_filter
from scipy.spatial import distance


# In[2]:


params = {'legend.fontsize': 50,
          'figure.figsize': (45, 15),
         'axes.labelsize': 60,
         'axes.titlesize':70,
         'xtick.labelsize':50,
         'ytick.labelsize':50,
         'lines.linewidth': 5}
plt.rcParams.update(params)


# In[5]:


def mapCodeSignal(Signal, Code, padding, startCycleCut = 0):
    cycleOn = 0
    
    listCodes = []
    
    ## Information on the possible codes
    ## organized as [name, cycles, colorOnGraph, IfUsedOnce]
    listCodes.append(["add", 1, "red", 0])
    listCodes.append(["adiw", 2, "silver", 0])
    listCodes.append(["and", 1, "tab:Brown", 0])
    listCodes.append(["asr", 1, "rosybrown", 0])
    listCodes.append(["bclr", 1, "lightcoral", 0])
    listCodes.append(["bld", 1, "indianred", 0])
    listCodes.append(["brbs-f", 1, "brown", 0])
    listCodes.append(["brbs-t", 2, "bisque", 0])
    listCodes.append(["brcs-f", 1, "darkorange", 0])
    listCodes.append(["brcs-t", 2, "burlywood", 0])
    listCodes.append(["break", 1, "tan", 0])
    listCodes.append(["breq-f", 1, "peru", 0])
    listCodes.append(["breq-t", 2, "firebrick", 0])
    listCodes.append(["brge-f", 1, "olive", 0])
    listCodes.append(["brge-t", 2, "grey", 0])
    listCodes.append(["brmi-f", 1, "navajowhite", 0])
    listCodes.append(["brmi-t", 2, "orange", 0])
    listCodes.append(["brne-f", 1, "magenta", 0]) 
    listCodes.append(["brne-t", 2, "orchid", 0])
    listCodes.append(["brpl-f", 1, "gold", 0])
    listCodes.append(["brpl-t", 2, "goldenrod", 0])
    listCodes.append(["brsh-f", 1, "darkgoldenrod", 0])
    listCodes.append(["brsh-t", 2, "khaki", 0])
    listCodes.append(["bset", 1, "darkkhaki", 0])
    listCodes.append(["bst", 1, "forestgreen", 0])
    listCodes.append(["call", 5, "aqua", 0])
    listCodes.append(["cbi", 2, "limegreen", 0])
    listCodes.append(["cbr", 1, "lime", 0])
    listCodes.append(["clc", 1, "aquamarine", 0])
    listCodes.append(["clh", 1, "tab:olive", 0])
    listCodes.append(["cli", 1, "lightseagreen", 0])
    listCodes.append(["cln", 1, "darkorchid", 0])
    listCodes.append(["clr", 1, "salmon", 0])
    listCodes.append(["cls", 1, "dimgrey", 0])
    listCodes.append(["clt", 1, "deeppink", 0])
    listCodes.append(["clv", 1, "fuchsia", 0])
    listCodes.append(["clz", 1, "salmon", 0])
    listCodes.append(["com", 1, "royalblue", 0])
    listCodes.append(["cp", 1, "y", 0])
    listCodes.append(["cpse-f",  1, "mediumaquamarine", 0])
    listCodes.append(["cpse-t-1", 2, "darkslategray", 0])
    listCodes.append(["cpse-t-2", 3, "darkslategrey", 0])
    listCodes.append(["dec", 1, "dimgray", 0])
    listCodes.append(["elpm", 3, "teal", 0])
    listCodes.append(["eor", 1, "purple", 0])
    listCodes.append(["in", 1, "darkcyan", 0])
    listCodes.append(["inc", 1, "yellow", 0])
    listCodes.append(["ld", 2, "slategrey", 0])
    listCodes.append(["ld+", 2, "slategrey", 0])
    listCodes.append(["ld-", 2, "cornflowerblue", 0])
    listCodes.append(["ldd", 2, "royalblue", 0])
    listCodes.append(["ldi", 1, "blue", 0])
    listCodes.append(["lds", 2, "navy", 0])
    listCodes.append(["lpm", 3, "midnightblue", 0])
    listCodes.append(["lsl", 1, "turquoise", 0])
    listCodes.append(["lsr", 1, "darkviolet", 0])
    listCodes.append(["mov", 1, "springgreen", 0])
    listCodes.append(["mul", 2, "darkgray", 0])
    listCodes.append(["fmul", 2, "purple", 0])
    listCodes.append(["neg", 1 , "darkmagenta", 0])
    listCodes.append(["nop", 1, "green", 0])
    listCodes.append(["or", 1, "black", 0])
    listCodes.append(["out", 1, "darkblue", 0])
    listCodes.append(["pop", 2, "mediumblue", 0])
    listCodes.append(["push", 2, "rebeccapurple", 0])
    listCodes.append(["jmp", 3, "cyan", 0])
    listCodes.append(["rcall", 4, "blueviolet", 0])
    listCodes.append(["ret", 5, "indigo", 0])
    listCodes.append(["rol", 1, "darkorchid", 0])
    listCodes.append(["ror", 1, "darkviolet", 0])
    listCodes.append(["rjmp", 2, "steelblue", 0])
    listCodes.append(["rcall", 3, "steelblue", 0])
    listCodes.append(["sbci-f", 1, "darkgreen", 0])
    listCodes.append(["sbci-t", 2, "purple", 0])
    listCodes.append(["sbi", 2, "orange", 0])
    listCodes.append(["sbiw", 2, "violet", 0])
    listCodes.append(["sbr", 1, "purple", 0])
    listCodes.append(["sbrc-f", 1, "deeppink", 0])
    listCodes.append(["sbrc-t-1", 2, "crimson", 0])
    listCodes.append(["sbrc-t-2", 3, "dodgerblue", 0])
    listCodes.append(["sec", 1, "peru", 0])
    listCodes.append(["sen", 1, "tomato", 0])
    listCodes.append(["ser", 1, "orange", 0])
    listCodes.append(["ses", 1, "deepskyblue", 0])
    listCodes.append(["sev", 1, "olivedrab", 0])
    listCodes.append(["sez", 1, "magenta", 0])
    listCodes.append(["sleep", 1, "saddlebrown", 0])
    listCodes.append(["spm", 4, "orangered", 0])
    listCodes.append(["st", 2, "tomato", 0])
    listCodes.append(["st+", 2, "tomato", 0])
    listCodes.append(["st-", 2, "greenyellow", 0])
    listCodes.append(["std", 2, "cadetblue", 0])
    listCodes.append(["sts", 2, "darkcyan", 0])
    listCodes.append(["sub", 1, "lawngreen", 0])
    listCodes.append(["swap", 1, "steelblue", 0])
    listCodes.append(["tst", 1, "blue", 0])
    listCodes.append(["wdr", 1, "deepskyblue", 0])
    
    found = 0
    codeNum = 0
    for code in Code:
        found = 0
        codeNum = 0
        
        while found == 0:
            if code.lower() == listCodes[codeNum][0]:
                found = 1
                if listCodes[codeNum][3] == 0:
                    if cycleOn == 0:
                        mapGraphTool(Signal, padding, cycleOn, listCodes[codeNum][1]-startCycleCut, listCodes[codeNum][0], 
                             listCodes[codeNum][2],  True)
                    else:
                        mapGraphTool(Signal, padding, cycleOn, listCodes[codeNum][1], listCodes[codeNum][0], 
                             listCodes[codeNum][2],  True)
                    listCodes[codeNum][3] = 1
                else:
                    mapGraphTool(Signal, padding, cycleOn, listCodes[codeNum][1], listCodes[codeNum][0], 
                             listCodes[codeNum][2])
                if cycleOn == 0:
                    cycleOn = cycleOn + listCodes[codeNum][1]-startCycleCut
                else:
                    cycleOn = cycleOn + listCodes[codeNum][1]
            else:
                codeNum = codeNum+1
            if codeNum > len(listCodes)-1:
                print(code + " is not a code name or not defined yet.")
                found = 1


# In[4]:


def mapGraphTool(Signal, padding, cycleStart, Cycles, name, cName, ifLabel=False):
    
    peakLocStart = cycleStart * 2
    if cycleStart == 0:
        seqStart = 0
    else:
        seqStart = getPeaksLoc(Signal, padding, peakLocStart)
    
    peakLocEnd = peakLocStart + (Cycles * 2)
    if seqStart > getPeaks(Signal, padding).shape[0]:
        seqEnd = Signal.shape[0]
    else:
        seqEnd = getPeaksLoc(Signal, padding, peakLocEnd)
    
    if ifLabel:
        plt.plot(np.arange(seqStart,seqEnd), Signal[seqStart:seqEnd], color = cName, label = name)
    else:
        plt.plot(np.arange(seqStart,seqEnd), Signal[seqStart:seqEnd], color = cName)


# In[ ]:


def getPeaksDataset(dataset, pad):
    dataset_Peaks = []

    for signal in dataset:
        temp = getPeaks(signal, pad)
        dataset_Peaks.append(temp)

    dataset_Peaks = np.array(dataset_Peaks, dtype=object)

    return dataset_Peaks


# In[ ]:


'''
Obtains the sequence from the startpeak to the end peak. Can minus indexes from the peaks found.
Signal: the given signal to cut the sequence from
pad: The amount of indexs to complete one cycle, also known as sampling rate
startPeak: The first peak to cut the sequence from
EndPeak: The final peak to cut the sequence from
minus: Number of indexs to move the cut to.
'''
# Obtains the sequence from the startpeak to the end peak. Can minus indexes from the peaks found.

def getSequence(signal, pad, startPeak, endPeak, minus):
    temp = signal[getPeaksLoc(signal, pad, startPeak)-minus:getPeaksLoc(signal, pad, endPeak)-minus]
    return temp


# In[1]:


'''
Obtains the sequence from the startpeak to the end peak. Can minus indexes from the peaks found.
Signals: the given signals to cut the sequence from
pad: The amount of indexs to complete one cycle, also known as sampling rate
startPeak: The first peak to cut the sequence from
EndPeak: The final peak to cut the sequence from
minus: Number of indexs to move the cut to.
'''
def getSequencesDataset(signals, pad, startPeak, endPeak, minus):
    tempsignals = []
    for signal in signals:
        temp = getSequence(signal, pad, startPeak, endPeak, minus)
        temp = np.array(temp)
        tempsignals.append(temp)
    tempsignals = np.array(tempsignals)
    return tempsignals


# In[ ]:


'''
Obtains the length determined across all signals for the sequence. Can minus indexes from the peaks found.
Signals: the given signals to cut the sequence from
pad: The amount of indexs to complete one cycle, also known as sampling rate
startPeak: The first peak to cut the sequence from
EndPeak: The final peak to cut the sequence from
minus: Number of indexs to move the cut to.
'''
def getSequencesDatasetFindLength(signals, pad, startPeak, endPeak, minus):
    tempsignals = []
    for signal in signals:
        temp = getSequence(signal, pad, startPeak, endPeak, minus)
        tempsignals.append(temp)
    tempsignals = np.array(tempsignals)
    
    tempSize = []
    for i in tempsignals:
        temp = i.shape[0]
        tempSize.append(temp)
    tempSize = np.array(tempSize)
    print(np.unique(tempSize))


# In[ ]:


'''
Obtains the length determined across all signals for the sequence. Can minus indexes from the peaks found.
Signals: the given signals to cut the sequence from
pad: The amount of indexs to complete one cycle, also known as sampling rate
startPeak: The first peak to cut the sequence from
EndPeak: The final peak to cut the sequence from
minus: Number of indexs to move the cut to.
'''
def getSequencesDatasetLength(signals, pad, startPeak, endPeak, length, minus):
    tempsignals = []
    for signal in signals:
        temp = getSequence(signal, pad, startPeak, endPeak, minus)
        temp = temp[:length]
        temp = np.array(temp)
        tempsignals.append(temp)
    tempsignals = np.array(tempsignals)
    tempsignals = np.reshape(tempsignals, (signals.shape[0], length))
    return tempsignals


# In[1]:


'''
def getPeaks(signal, padding, high = True):
    loc = 0
    listPeaks = []
    goingUP = True
    lastHLoc = 0 - padding*2
    newHLoc = 0 
    
    # set if going up or down.
    lastLLoc = (padding*2) *-1
    if loc == 0:
        if signal[loc+1] > signal[loc]:
            goingUP = True
        else:
            goingUP = False

    while loc < signal.shape[0]-1:            
        # Check if next is lower when going up.
        if goingUP:
            if signal[loc+1] < signal[loc]:
                # Check next few to see if higher point.
                #TrueHigh = False
                count = 0
                while count < padding:
                    if loc + count < signal.shape[0]:
                        if signal[loc] < signal[loc + count]:
                            loc = loc + count
                            count = padding  
                    count = count+1
                    #if count == padding:
                        #TrueHigh = True
                #if TrueHigh:     
                newHLoc = loc
                # print(newHLoc)

                if lastHLoc + padding < newHLoc:
                    if high == True:
                        listPeaks.append(signal[loc])
                    lastHLoc = newHLoc
                goingUP = False
            else:
                loc = loc+1
            
        # Check if next is higher when going down.
        elif goingUP == False:
            if signal[loc+1] > signal[loc]:
                # Check next few to see if higher point.
                #TrueLow = False
                count = 0
                while count < padding:
                    if loc + count < signal.shape[0]-1:
                        if signal[loc] > signal[loc + count]:
                            loc = loc + count
                            count = padding             
                    count = count+1
                    #if count == padding:
                        #TrueLow = True
                #if TrueLow:
                newLLoc = loc
                # print(newHLoc)

                if lastLLoc + padding < newLLoc:
                    if high == False:
                        listPeaks.append(signal[loc])
                    lastLLoc = newLLoc
                goingUP = True
            else:
                loc = loc+1
        
    # Check last value.
    if goingUP:
        if signal.shape[0]-1 > lastHLoc + padding:
            listPeaks.append(signal[signal.shape[0]-1])
    
    return np.array(listPeaks)
'''


# In[ ]:


def getPeaks(signal, padding, getHigh = True):
    loc = 0
    listPeaks = []
    goingUP = True
    lastHLoc = 0 - padding*2
    newHLoc = 0 
    
    # set if going up or down.
    ## Determine the mid point.
    cutpoint = signal.min() + ((signal.max() - signal.min()) / 2)
    
    lastLLoc = (padding*2) *-1
    if loc == 0:
        if signal[loc] > cutpoint:
            goingUP = True
        else:
            goingUP = False

    while loc < signal.shape[0]-1:            
        # Check if next is lower when going up.
        if goingUP:
            if signal[loc+1] < signal[loc]:
                # Check next few to see if higher point.
                TrueHigh = False
                count = 0
                while count < padding:
                    if loc + count < signal.shape[0]:
                        if signal[loc] < signal[loc + count]:
                            loc = loc + count
                            count = 0 
                    count = count+1
                    if count == padding:
                        TrueHigh = True
                if TrueHigh:     
                    newHLoc = loc
                    # print(newHLoc)
                    if lastHLoc + padding < newHLoc:
                        if getHigh == True:
                            listPeaks.append(signal[loc])
                        lastHLoc = newHLoc
                    goingUP = False
            else:
                loc = loc+1
            
        # Check if next is higher when going down.
        elif goingUP == False:
            if signal[loc+1] > signal[loc]:
                # Check next few to see if higher point.
                TrueLow = False
                count = 0
                while count < padding:
                    if loc + count < signal.shape[0]-1:
                        if signal[loc] > signal[loc + count]:
                            loc = loc + count
                            count = 0             
                    count = count+1
                    if count == padding:
                        TrueLow = True
                if TrueLow:
                    newLLoc = loc
                    if lastLLoc + padding < newLLoc:
                        if getHigh == False:
                            listPeaks.append(signal[loc])
                        lastLLoc = newLLoc
                    goingUP = True
            else:
                loc = loc+1
        
    # Check last value.
    if goingUP:
        if signal.shape[0]-1 > lastHLoc + padding:
            if getHigh == True:
                listPeaks.append(signal[signal.shape[0]-1])
    
    elif goingUP == False:
        if signal.shape[0]-1 > lastLLoc + padding:
            if getHigh == False:
                listPeaks.append(signal[signal.shape[0]-1])
    
    return np.array(listPeaks)


# In[11]:


def getPeaksLoc(signal, padding, peakNum, getHigh = True):
    loc = 0
    listPeaks = []
    goingUP = True
    peak = -1
    lastHLoc = 0 - padding*2
    lastLLoc = 0 
    
    # set if going up or down.
    ## Determine the mid point.
    cutpoint = signal.min() + (signal.max() - signal.min() / 2)
    
    lastLLoc = (padding*2) *-1
    if loc == 0:
        if signal[loc] > cutpoint:
            goingUP = True
        else:
            goingUP = False

    while loc < signal.shape[0]-1:              
        # Check if next is lower when going up.
        if goingUP:
            if signal[loc+1] < signal[loc]:
                # Check next few to see if higher point.
                TrueHigh = False
                count = 0
                while count < padding:
                    if loc + count < signal.shape[0]:
                        if signal[loc] < signal[loc + count]:
                            loc = loc + count
                            count = 0             
                    count = count+1
                    if count == padding:
                        TrueHigh = True
                if TrueHigh:
                    newHLoc = loc
                    
                    if lastHLoc + padding < newHLoc:
                        if getHigh == True:
                            listPeaks.append(signal[loc])
                            peak = peak + 1
                            if peak == peakNum:
                                return loc
                        lastHLoc = newHLoc
                    goingUP = False
            else:
                loc = loc+1
            
        # Check if next is higher when going down.
        elif goingUP == False:
            if signal[loc+1] > signal[loc]:
                # Check next few to see if higher point.
                TrueLow = False
                count = 0
                while count < padding:
                    if loc + count < signal.shape[0]-1:
                        if signal[loc] > signal[loc + count]:
                            loc = loc + count
                            count = 0             
                    count = count+1
                    if count == padding:
                        TrueLow = True
                if TrueLow:
                    newLLoc = loc
                    if lastLLoc + padding < newLLoc:
                        if getHigh == False:
                            listPeaks.append(signal[loc])
                            peak = peak + 1
                            if peak == peakNum:
                                return loc
                        lastLLoc = newLLoc
                    goingUP = True
            else:
                loc = loc+1
    
    # Check last value.
    # Check last value.
    if goingUP:
        if signal.shape[0]-1 > lastHLoc + padding:
            if getHigh == True:
                listPeaks.append(signal[signal.shape[0]-1])
    
    elif goingUP == False:
        if signal.shape[0]-1 > lastLLoc + padding:
            if getHigh == False:
                listPeaks.append(signal[signal.shape[0]-1])
    
    return signal.shape[0]-1


# In[ ]:




