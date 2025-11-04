#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# File contains code to extract sequences of instructions from a program.


# In[ ]:


from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import random
import tensorflow as tf
import numpy as np
import timeit
from sklearn import metrics
import scipy.io as sio
import csv
import os
import math 
import random as rand
from numpy import genfromtxt
from numpy import sum,isrealobj,sqrt
from numpy.random import standard_normal

#import torch
#from torch import nn

import warnings
from scipy.spatial import distance

import sys

from Modules.Tools import MapTool
from Modules.Tools import BoxPlot
from Modules.Tools import peakCorrelation as pC
from Modules.Tools import ATmega2560_Instructions as Inst_Info


# In[ ]:


def getValue(dataset, loc):
    list_peaks = []
    
    for peaks in dataset:
        list_peaks.append(peaks[loc])
    return np.array(list_peaks)


# In[1]:


'''
Simply splits known multi cycle codes.
'''
def splitCode(array):
    code_list = []
    two_cycle_list = ["adiw", "sbiw", "sts", "lds", "muls", "fmul", 
                      "push", "pop", "rjmp"]
    three_cycle_list = ["jmp"]
    four_cycle_list = []
    five_cycle_list = ["call", "ret"]
    for code in array:
        # check for hyphens
        if code == "brne-f":
            code_list.append("brneF")
        elif code == "brne-t":
            code_list.append("brneTPT1")
            code_list.append("brneTPT2")
        elif code == "breq-f":
            code_list.append("breqF")
        elif code == "breq-t":
            code_list.append("breqTPT1")
            code_list.append("breqTPT2")
        elif code == "brpl-f":
            code_list.append("brplF")
        elif code == "brpl-t":
            code_list.append("brplTPT1")
            code_list.append("brplTPT2")
        elif code == "sbis-f":
            code_list.append("sbisF")
        elif code == "sbis-t":
            code_list.append("sbisTPT1")
            code_list.append("sbisTPT2")
        
        # check for two cycles
        elif code in two_cycle_list:
            code_list.append(code + "PT1")
            code_list.append(code + "PT2")
            
        # check for three cycles
        elif code in three_cycle_list:
            code_list.append(code + "PT1")
            code_list.append(code + "PT2")
            code_list.append(code + "PT3")
        
        # check for four cycles
        elif code in four_cycle_list:
            code_list.append(code + "PT1")
            code_list.append(code + "PT2")
            code_list.append(code + "PT3")
            code_list.append(code + "PT4")
            
        # check for five cycles
        elif code in five_cycle_list:
            code_list.append(code + "PT1")
            code_list.append(code + "PT2")
            code_list.append(code + "PT3")
            code_list.append(code + "PT4")
            code_list.append(code + "PT5")
        
        else:
            code_list.append(code)
            
    return np.array(code_list)


# In[ ]:


'''
Return a dictionary of (A) datapoints from the end of the previous to the start of the next instruction surrounding each sequence
for all signals and (B) the names of the instuctions being executed in the sequence. I.E. {Data: [], Code: []}

Parameters
Signals: the signals to extract information from.
peakStart: the starting peak to start extracting information from.
peakEnd: the ending peak to extract information from.
code: string assembly code that is in the total sequence of code to extract from. Given in 1D numpy array
amount: amount of code to extract each time.
length: the amount of time indexs per sequence. (this is due to the fact that a certain limit 
will be necessary for the NN model)
skip: amount of code to skip each time. Default = 1.
'''
def extractSequences(signals, peakStart, peakEnd, code, amount, length, pad, skip=1):
    # list object to contain the dictionry information.
    listOfInfo = []
    # cycle through the code sequences to extract
    for c in range(0, code.shape[0]-(amount-1), skip):
        # obtain the ASM code for the sequence.
        ## list object to contain the exact code of the sequence.
        seqCode =[]
        ## cycle through the amount of code in the sequence
        for i in range(amount):
            seqCode.append(code[c+i])
        ## Convert to numpy
        seqCode = np.array(seqCode)
        ## Obtain the amount of cycles in the sequenceCode
        ### Note for now we will use just the 1 cycle (2 peaks) for each instruction as this is an experiment.
        ### Later will need to identify which instructions cost more cycles and perform a lookup.
        amtOfCycles = 1*amount
        amtOfPeaks = amtOfCycles * 2
        amtOfPrior = 2
        step = c * (1*2)
        
        # Obtain the extracted sequences from the signals.
        seqs = []
        ## cycle through each signal
        for s in range(signals.shape[0]):
            ## extract the sequence
            ## This sequence will be from the low peak before the sequence to the high peak after the sequence.
            seq = signals[s][MapTool.getPeaksLoc(signals[s], pad, step+peakStart, False): MapTool.getPeaksLoc(signals[s], pad, step+peakStart+amtOfPeaks)]
            ## now that we have the sequnce, we need to cut equally from both ends to the size necessary,
            ## This will give the best chance to avoid removing data from what we will be extracting from the
            ## synthetic example. I.E. from 0 value before and after code sequence.
            seq = refitSignal(seq, length)
            seqs.append(seq)
        
        seqs = np.array(seqs)
        # Create a dictionary item pair of code and sequecnes.
        dictPair = {"Seq": seqs, "Code": seqCode}
        listOfInfo.append(dictPair)
    
    return listOfInfo


# In[ ]:


'''
This method cuts the dataset down to the desired size by removing equal parts from both sides. 
(front will take out more if necessary).
signal: the signal to cut from.
length: the length that the signal needs to be.
'''
def refitSignal(signal, length):
    # Identify how many to remove
    currentSize = signal.shape[0]
    removeCount = currentSize - length
    # Check to see if it is possible to remove that many.
    if removeCount < 0:
        print("ERROR: size is set to be to large, readjust to at least: " + str(currentSize))
        return signal
    else:
        # flip from removing beginning or end of signal to remove data from.
        front = True
        tempSignal = np.copy(signal)
        while removeCount > 0:
            if front == True:
                # take from beginning
                tempSignal = tempSignal[1:]
                removeCount = removeCount - 1
                front == False
            else:
                # take from end
                tempSignal = tempSignal[:tempSignal.shape[0]-2]
                removeCount = removeCount-1
                front == True
        return tempSignal


# In[ ]:


'''
This will give the list of lengths that is seen for the desired sequences.
This is used to determine the optimal length for the sequences which must be the same 
and all sequnces must be able to contain that much information.
'''
def extractSequencesFindLimit(signals, code, amount, pad, peakStart=0, skip=1):
    # list object to contain the dictionry information.
    listOfInfo = []
    # cycle through the code sequences to extract
    for c in range(0, code.shape[0]-(amount-1), skip):
        # obtain the ASM code for the sequence.
        ## list object to contain the exact code of the sequence.
        seqCode =[]
        ## cycle through the amount of code in the sequence
        for i in range(amount):
            seqCode.append(code[c+i])
        ## Convert to numpy
        seqCode = np.array(seqCode)
        ## Obtain the amount of cycles in the sequenceCode
        ### Note for now we will use just the 1 cycle (2 peaks) for each instruction as this is an experiment.
        ### Later will need to identify which instructions cost more cycles and perform a lookup.
        amtOfCycles = 1*amount
        amtOfPeaks = amtOfCycles * 2
        amtOfPrior = 2
        step = c * (1*2)
        
        tempLimits = []
        # Obtain the extracted sequences from the signals.
        ## cycle through each signal
        for s in range(signals.shape[0]):
            ## extract the sequence
            ## This sequence will be from the low peak before the sequence to the high peak after the sequence.
            seq = signals[s][MapTool.getPeaksLoc(signals[s], pad, step+peakStart, False): MapTool.getPeaksLoc(signals[s], pad, step+peakStart+amtOfPeaks)]
            #print(seq)
            ## now that we have the sequnce, we need to cut equally from both ends to the size necessary,
            ## This will give the best chance to avoid removing data from what we will be extracting from the
            ## synthetic example. I.E. from 0 value before and after code sequence.
            tempLimit = seq.shape[0]
            #if tempLimit < 135:
                #print(s)
            tempLimits.append(tempLimit)
        
        tempLimits = np.array(tempLimits)
        # Create a dictionary item pair of code and sequecnes.
        #print("The unique lengths for sequence: ", seqCode)
        #print(np.unique(tempLimits))
        #print(tempLimits)
        listOfInfo.append(tempLimits)
    
    listOfInfo = np.array(listOfInfo)
    listOfInfo = listOfInfo.flatten()
    
    return listOfInfo


# # Now we have a method that will extract sequences, however many of the sequences obtained are repeats between other datasets.
# 
# - Create a method that will take all sets of EM signals and codes
#     - extract the sequences and codes from each
#     - Identify and combine like code
#     - Return the code and sequences across all sets.

# In[ ]:


'''
This method will extract the sequences and code of the sequences for multiple sets of signals.
It will also combine the like code sequences together.

Parameters
signalSets: the signal sets given in a 3D numpy matrix
Codes: full code of the extrancting sequences for the signals, given in a 2D numpy matrix
peakStarts: the starting peak for each sets EM signal to extract from, given in a 2D numpy matrix
peakEnds: the ending peak for each sets EM signal to extract from, given in a 2D numpy matrix
amount: the amount of instructions per sequence.
skip: the amount to of instructions to skip by per sequence extraction.
'''
def extractSequencesDataset(signalSets, Codes, peakStarts, peakEnds, amount, pad, skip=1, limit=0):
    # create a list of entire sequence information
    Info = []
    
    if limit == 0:
        print("Finding limit...")

        minilimit = 9999999999
        # identify the limit
        ## Obtain the information for one set
        for n in range(Codes.shape[0]):
            print("Working on Program: ", n)
            signals = signalSets[n]
            code = Codes[n]
            peakStart = peakStarts[n]
            peakEnd = peakEnds[n]

            # obtain the optimal limit for the set
            limits = extractSequencesFindLimit(signals, peakStart, peakEnd, code, amount)
            limit = limits.min()
            # Identify the best limit across all sets
            if limit < minilimit:
                minilimit = limit

        print("Limit Found! ", minilimit)
    
    else:
        minilimit = limit
        print("Limit set to: ", minilimit)
    
    print("Extracting sequences...")
    # identify the sequences and sequence codes.
    ## Obtain the information for one set
    for n in range(Codes.shape[0]):
        print("Working on Program: ", n)
        signals = signalSets[n]
        code = Codes[n]
        peakStart = peakStarts[n]
        peakEnd = peakEnds[n]
        
        # obtain the sequence and code sequences information for the set
        tempInfo = extractSequences(signals, peakStart, peakEnd, code, amount, minilimit)
        
        # cycle through each obtained info
        for tempinfo in tempInfo:
            #print(tempinfo["Code"])
            # check if Info is empty
            if len(Info) == 0:
                # add the instances of the obtained info to Info
                dictSeq = {"Seq": tempinfo["Seq"], "Code": tempinfo["Code"]}
                Info.append(dictSeq)
                #print("Check")
            else:
                # check if there is already an occures of the sequence code.
                foundInst = False
                for info in Info:
                    if np.array_equal(info["Code"], tempinfo["Code"]):
                        # concatentated the two sets of sequences as they are for the same code sequence.
                        info["Seq"] = np.concatenate((info["Seq"], tempinfo["Seq"]))
                        foundInst = True
                
                # if no already occures in Info list then add.
                if foundInst == False:
                    #print("Not Found")
                    # add the instances of the obtained info to Info
                    dictSeq = {"Seq": tempinfo["Seq"], "Code": tempinfo["Code"]}
                    Info.append(dictSeq)
                    
    return Info


# # Code for getting only the peaks

# In[ ]:


'''
Return a dictionary of (A) datapoints from the end of the previous to the start of the next instruction surrounding each sequence
for all signals and (B) the names of the instuctions being executed in the sequence. I.E. {Data: [], Code: []}

Parameters
Signals: the signals to extract information from.
peakStart: the starting peak to start extracting information from.
peakEnd: the ending peak to extract information from.
code: string assembly code that is in the total sequence of code to extract from. Given in 1D numpy array
amount: amount of code to extract each time.
length: the amount of time indexs per sequence. (this is due to the fact that a certain limit will be necessary for the NN model)
skip: amount of code to skip each time. Default = 1.
'''
def extractSequences_Peaks(signals, peakStart, peakEnd, code, amount, length, pad, skip=1):
    # list object to contain the dictionry information.
    listOfInfo = []
    # cycle through the code sequences to extract
    for c in range(0, code.shape[0]-(amount-1), skip):
        # obtain the ASM code for the sequence.
        ## list object to contain the exact code of the sequence.
        seqCode =[]
        ## cycle through the amount of code in the sequence
        for i in range(amount):
            seqCode.append(code[c+i])
        ## Convert to numpy
        seqCode = np.array(seqCode)
        ## Obtain the amount of cycles in the sequenceCode
        ### Note for now we will use just the 1 cycle (2 peaks) for each instruction as this is an experiment.
        ### Later will need to identify which instructions cost more cycles and perform a lookup.
        amtOfCycles = 1*amount
        amtOfPeaks = amtOfCycles * 2
        amtOfPrior = 2
        step = c * (1*2)
        
        # Obtain the extracted sequences from the signals.
        seqs = []
        ## cycle through each signal
        for s in range(signals.shape[0]):
            ## extract the sequence
            ## This sequence will be from the low peak before the sequence to the high peak after the sequence.
            seq = signals[s][MapTool.getPeaksLoc(signals[s], pad, step+peakStart, False): MapTool.getPeaksLoc(signals[s], pad, step+peakStart+amtOfPeaks-1)+1]
            ## now that we have the sequnce, we need to cut equally from both ends to the size necessary,
            ## This will give the best chance to avoid removing data from what we will be extracting from the
            ## synthetic example. I.E. from 0 value before and after code sequence.
            seq = MapTool.getPeaks(seq, pad)
            seqs.append(seq)
        
        seqs = np.array(seqs)
        # Create a dictionary item pair of code and sequecnes.
        dictPair = {"Seq": seqs, "Code": seqCode}
        listOfInfo.append(dictPair)
    
    return listOfInfo


# In[ ]:


'''
This method will extract the sequences and code of the sequences for multiple sets of signals.
It will also combine the like code sequences together.
This will get only the peaks

Parameters
signalSets: the signal sets given in a 3D numpy matrix
Codes: full code of the extrancting sequences for the signals, given in a 2D numpy matrix
peakStarts: the starting peak for each sets EM signal to extract from, given in a 2D numpy matrix
peakEnds: the ending peak for each sets EM signal to extract from, given in a 2D numpy matrix
amount: the amount of instructions per sequence.
skip: the amount to of instructions to skip by per sequence extraction.
'''
def extractSequencesDataset_Peaks(signalSets, Codes, peakStarts, peakEnds, amount, skip=1, limit=0):
    # create a list of entire sequence information
    Info = []
    
    if limit == 0:
        print("Finding limit...")

        minilimit = 9999999999
        # identify the limit
        ## Obtain the information for one set
        for n in range(Codes.shape[0]):
            print("Working on Program: ", n)
            signals = signalSets[n]
            code = Codes[n]
            peakStart = peakStarts[n]
            peakEnd = peakEnds[n]

            # obtain the optimal limit for the set
            limits = extractSequencesFindLimit(signals, peakStart, peakEnd, code, amount)
            limit = limits.min()
            # Identify the best limit across all sets
            if limit < minilimit:
                minilimit = limit

        print("Limit Found! ", minilimit)
    
    else:
        minilimit = limit
        print("Limit set to: ", minilimit)
    
    print("Extracting sequences...")
    # identify the sequences and sequence codes.
    ## Obtain the information for one set
    for n in range(Codes.shape[0]):
        print("Working on Program: ", n)
        signals = signalSets[n]
        code = Codes[n]
        peakStart = peakStarts[n]
        peakEnd = peakEnds[n]
        
        # obtain the sequence and code sequences information for the set
        tempInfo = extractSequences_Peaks(signals, peakStart, peakEnd, code, amount, minilimit)
        
        # cycle through each obtained info
        for tempinfo in tempInfo:
            #print(tempinfo["Code"])
            # check if Info is empty
            if len(Info) == 0:
                # add the instances of the obtained info to Info
                dictSeq = {"Seq": tempinfo["Seq"], "Code": tempinfo["Code"]}
                Info.append(dictSeq)
                #print("Check")
            else:
                # check if there is already an occures of the sequence code.
                foundInst = False
                for info in Info:
                    if np.array_equal(info["Code"], tempinfo["Code"]):
                        # concatentated the two sets of sequences as they are for the same code sequence.
                        info["Seq"] = np.concatenate((info["Seq"], tempinfo["Seq"]))
                        foundInst = True
                
                # if no already occures in Info list then add.
                if foundInst == False:
                    #print("Not Found")
                    # add the instances of the obtained info to Info
                    dictSeq = {"Seq": tempinfo["Seq"], "Code": tempinfo["Code"]}
                    Info.append(dictSeq)
                    
    return Info


# # Code for getting only the Major peak of the final instruction

# In[2]:


'''
Return a dictionary of (A) datapoints from the end of the previous to the start of the next instruction surrounding each sequence
for all signals and (B) the names of the instuctions being executed in the sequence. I.E. {Data: [], Code: []}

Parameters
Signals: the signals to extract information from.
peakStart: the starting peak to start extracting information from.
peakEnd: the ending peak to extract information from.
code: string assembly code that is in the total sequence of code to extract from. Given in 1D numpy array
amount: amount of code to extract each time.
skip: amount of code to skip each time. Default = 1.
'''
def extractSequences_MajorPeak_Old(signals, peakStart, peakEnd, code, amount, pad, skip=1):
    # list object to contain the dictionry information.
    listOfInfo = []
    # cycle through the code sequences to extract
    for c in range(0, code.shape[0]-(amount-1), skip):
        # obtain the ASM code for the sequence.
        ## list object to contain the exact code of the sequence.
        seqCode =[]
        ## cycle through the amount of code in the sequence
        for i in range(amount):
            seqCode.append(code[c+i])
        ## Convert to numpy
        seqCode = np.array(seqCode)
        ## Obtain the amount of cycles in the sequenceCode
        ### Note for now we will use just the 1 cycle (2 peaks) for each instruction as this is an experiment.
        ### Later will need to identify which instructions cost more cycles and perform a lookup.
        amtOfCycles = 1*amount
        amtOfPeaks = amtOfCycles * 2
        amtOfPrior = 2
        step = c * (1*2)
        
        # Obtain the extracted sequences from the signals.
        seqs = []
        ## cycle through each signal
        for s in range(signals.shape[0]):
            ## extract the sequence
            ## This sequence will be from the low peak before the sequence to the high peak after the sequence.
            seq = signals[s][MapTool.getPeaksLoc(signals[s], pad, step+peakStart, False): MapTool.getPeaksLoc(signals[s], pad, step+peakStart+amtOfPeaks-1)+1]
            ## now that we have the sequnce, we need to cut equally from both ends to the size necessary,
            ## This will give the best chance to avoid removing data from what we will be extracting from the
            ## synthetic example. I.E. from 0 value before and after code sequence.
            seq = MapTool.getPeaks(seq, pad)
            # Note we need the second to last. last is the second peak in the cycle.
            # Also the peak amounts is from 0-total-1. So its -2
            majorpeak = np.array([seq[seq.shape[0]-2]])
            seqs.append(majorpeak)
        
        seqs = np.array(seqs)
        # Create a dictionary item pair of code and sequecnes.
        dictPair = {"Seq": seqs, "Code": seqCode}
        listOfInfo.append(dictPair)
    
    return listOfInfo


# In[ ]:


'''
Return a dictionary of (A) datapoints from the end of the previous to the start of the next instruction surrounding each sequence
for all signals and (B) the names of the instuctions being executed in the sequence. I.E. {Data: [], Code: []}

Parameters
Signals: the signals to extract information from, Assumes that the provided info is just the MajorPeaks.
peakStart: the starting peak to start extracting information from.
peakEnd: the ending peak to extract information from.
code: string assembly code that is in the total sequence of code to extract from. Given in 1D numpy array
amount: amount of code to extract each time.
skip: amount of code to skip each time. Default = 1.
'''
def extractSequences_MajorPeak(Signals, peakStart, peakEnd, code, amount, pad, skip=1):
    start_time = timeit.default_timer()
    
    signals = MapTool.getPeaksDataset(Signals, pad)
    # list object to contain the dictionry information.
    listOfInfo = []
    # cycle through the code sequences to extract
    for c in range(0, code.shape[0]-(amount-1), skip):
        # obtain the ASM code for the sequence.
        ## list object to contain the exact code of the sequence.
        seqCode =[]
        ## cycle through the amount of code in the sequence
        for i in range(amount):
            seqCode.append(code[c+i])
        ## Convert to numpy
        seqCode = np.array(seqCode)
        ## Obtain the amount of cycles in the sequenceCode
        ### Note for now we will use just the 1 cycle (2 peaks) for each instruction as this is an experiment.
        ### Later will need to identify which instructions cost more cycles and perform a lookup.
        amtOfCycles = 1*amount
        amtOfPeaks = amtOfCycles * 2
        amtOfPrior = 2
        step = c * (1*2)
        
        # Obtain the extracted sequences from the signals.
        seqs = []
        ## cycle through each signal
        for s in range(signals.shape[0]):
            # Note we need the second to last. last is the second peak in the cycle.
            # Also the peak amounts is from 0-total-1. So its -2
            majorpeak = [signals[s][step+peakStart]]
            seqs.append(majorpeak)
        
        seqs = np.array(seqs)
        # Create a dictionary item pair of code and sequecnes.
        dictPair = {"Seq": seqs, "Code": seqCode}
        listOfInfo.append(dictPair)
    
    end_time = timeit.default_timer()
    
    print("Time: ", end_time-start_time)
    
    return listOfInfo


# In[4]:


'''
This method will extract the sequences and code of the sequences for multiple sets of signals.
It will also combine the like code sequences together.
This will get only the peaks

Parameters
signalSets: the signal sets given in a 3D numpy matrix
Codes: full code of the extrancting sequences for the signals, given in a 2D numpy matrix
peakStarts: the starting peak for each sets EM signal to extract from, given in a 2D numpy matrix
peakEnds: the ending peak for each sets EM signal to extract from, given in a 2D numpy matrix
amount: the amount of instructions per sequence.
skip: the amount to of instructions to skip by per sequence extraction.
'''
def extractSequencesDataset_MajorPeak(signalSets, Codes, peakStarts, peakEnds, amount, pad, skip=1):
    # create a list of entire sequence information
    Info = []
    
    print("Extracting sequences...")
    # identify the sequences and sequence codes.
    ## Obtain the information for one set
    for n in range(Codes.shape[0]):
        print("Working on Program: ", n)
        signals = signalSets[n]
        code = Codes[n]
        peakStart = peakStarts[n]
        peakEnd = peakEnds[n]
        
        # obtain the sequence and code sequences information for the set
        tempInfo = extractSequences_MajorPeak(signals, peakStart, peakEnd, code, amount, pad=pad)
        
        # cycle through each obtained info
        for tempinfo in tempInfo:
            #print(tempinfo["Code"])
            # check if Info is empty
            if len(Info) == 0:
                # add the instances of the obtained info to Info
                dictSeq = {"Seq": tempinfo["Seq"], "Code": tempinfo["Code"]}
                Info.append(dictSeq)
                #print("Check")
            else:
                # check if there is already an occures of the sequence code.
                foundInst = False
                for info in Info:
                    if np.array_equal(info["Code"], tempinfo["Code"]):
                        # concatentated the two sets of sequences as they are for the same code sequence.
                        info["Seq"] = np.concatenate((info["Seq"], tempinfo["Seq"]))
                        foundInst = True
                
                # if no already occures in Info list then add.
                if foundInst == False:
                    #print("Not Found")
                    # add the instances of the obtained info to Info
                    dictSeq = {"Seq": tempinfo["Seq"], "Code": tempinfo["Code"]}
                    Info.append(dictSeq)
                    
    return Info


# In[5]:





# In[ ]:


'''
Converts the seq code information from assembly instruction into 
tokens from the given tokenizer.
'''
def convertASMToTokens(Info, tokenizer):
    Base = []
    for seq in Info:
        code = seq["Code"]
        tokens = tokenizer.texts_to_sequences(code)
        token_list = []
        for token in tokens:
            token_list.append(token[0])
        Base.append(token_list)
    return np.array(Base)


# In[ ]:


'''
New method to obtain the sequence code.
'''
def obtainSeqASMCode(ASM_Split, tokenizer, seq_size=5):
    seqs = []
    for i in range(ASM_Split.shape[0]-seq_size+1):
        seq = []
        for n in range(seq_size):
            seq.append(ASM_Split[i+n])
        seqs.append(seq)
    seqs_ASM = []
    for code in seqs:
        tokens = tokenizer.texts_to_sequences(code)
        token_list = []
        for token in tokens:
            token_list.append(token[0])
        seqs_ASM.append(token_list)
    return np.array(seqs), np.array(seqs_ASM)


# In[ ]:


'''
This will take the code and sequence information and convert it into training set to run GANs on.
'''
def convertToGANsTrainSet(Info, tokenizer, max_seq):
    first = True
    for info in Info:
        seqs = info["Seq"]
        code = info["Code"].tolist()
        code_tok = tokenizer.texts_to_sequences([code])
        print(code_tok)
        # full with null value to indicate cut off if code sequence is less thatn max_seq
        is_correct_length = False
        while is_correct_length == False:
            if len(code_tok[0]) == max_seq:
                is_correct_length = True
            else:
                null_code = [-1]
                code_tok[0] = null_code + code_tok[0]
        code_tok = np.array(code_tok)
        
        codes = np.tile(code_tok, (seqs.shape[0],1))
        
        if first:
            all_seq = np.array(seqs)
            all_code = np.array(codes)
            first = False
        
        else:
            all_seq = np.concatenate((all_seq, seqs))
            all_code = np.concatenate((all_code, codes))
    
    all_seq = np.array(all_seq)
    all_code = np.array(all_code)
    
    return all_seq, all_code


# In[ ]:


'''
This will take the code and sequence information and convert it into training set to run GANs on.
'''
def convertToGANsTrainSet_Balance(Info, tokenizer, max_seq, amount, random_seed=0):
    first = True
    for info in Info:
        seqs = np.copy(info["Seq"])
        np.random.seed(random_seed)
        random.shuffle(seqs)
        seqs = seqs[:amount]
        
        code = info["Code"].tolist()
        code_tok = tokenizer.texts_to_sequences([code])
        # full with null value to indicate cut off if code sequence is less thatn max_seq
        is_correct_length = False
        while is_correct_length == False:
            if len(code_tok[0]) == max_seq:
                is_correct_length = True
            else:
                null_code = [-1]
                code_tok[0] = null_code + code_tok[0]
        code_tok = np.array(code_tok)
        
        codes = np.tile(code_tok, (seqs.shape[0],1))
        
        if first:
            all_seq = np.array(seqs)
            all_code = np.array(codes)
            first = False
        
        else:
            all_seq = np.concatenate((all_seq, seqs))
            all_code = np.concatenate((all_code, codes))
    
    all_seq = np.array(all_seq)
    all_code = np.array(all_code)
    
    return all_seq, all_code


# # This is part of the version 2 methods.
# - Version 2 takes into account register values as well as the instruction sequence.

# In[ ]:


'''
Checks if string s is a hex number
'''
def is_hex(s):
    return all(c in string.hexdigits for c in s)


# In[ ]:


def isInst(word):
    if ";" in word or "/" in word:
        #print("Found a note")
        #print(word)
        return False
    elif ":" in word:
        #print("Found a jmp point")
        #print(word)
        return False
    else:
        return True


# In[ ]:


def actualRegOrValue(command, Command):
    new_command = command
    if "," in command:
        new_command = command[:len(command)-1]
    
    if "br" in new_command:
        if "TRUE" in Command or "TRUE." in Command:
            new_command = new_command + "-t"
        elif "FALSE" in Command or "FALSE." in Command:
            new_command = new_command + "-f"
            
    return new_command


# In[ ]:


def GetActualCommand(Command):
    actualCommand = []
    for command in Command:
        actualCommand.append(actualRegOrValue(command, Command))
        
    return actualCommand


# In[ ]:


def get2RegValue(v1, v2):
    v1_bin = bin(v1)[2:].zfill(8)
    v2_bin = bin(v2)[2:].zfill(8)
    
    #print(v1_bin)
    #print(v2_bin)
    
    #v1_ext = bin(v1<<8)
    ans = int(v1_bin + v2_bin,2)
    
    #print(v1_bin)
    #print(v2_bin)
    
    #print(bin(v1<<8)[2:])
    #print(ans)
    #print(int(ans, 2))
    return ans


# In[ ]:


def getRegs(command):
    r1 = ""
    r2 = ""
    if ":" in command:
        loc = 0
        for i in range(len(command)):
            if command[i] == ":":
                loc = i
        r1 = command[:loc]
        r2 = command[loc+1:]
        
    else:
        print("ERROR: no two registers seen.")
        
    return r1, r2


# In[ ]:


def Opt2Regs(Opt, v1, v2, value):
    maxnum = int('1111111111111111',2)
    v1_bin = bin(v1)[2:].zfill(8)
    v2_bin = bin(v2)[2:].zfill(8)
    
    #print(v1_bin)
    #print(v2_bin)
    
    #v1_ext = bin(v1<<8)
    ans = int(v1_bin + v2_bin,2)
    #print(ans)
    
    if Opt == "adiw":
        ans = ans + value
        if ans > maxnum:
            ans = ans - maxnum -1
    elif Opt == "sbiw":
        ans = ans - value
        if ans < -1:
            ans = maxnum + ans +1
    ans = bin(ans)[2:].zfill(16)
    #print(ans)
    
    r1 = int(ans[:8],2)
    r2 = int(ans[8:],2)
    
    #print(r1)
    #print(r2)
    
    return r1, r2


# In[ ]:


def getInt(value):
    if "x" in value:
        return int(value[2:])
    else:
        return int(value)


# In[ ]:


class Register:
    def __init__(self, name, value):
        self.name = name
        self.value = value

    def getName(self):
        return self.name
    
    def getValue(self):
        return self.value
    
    def changeValue(self, new_value):
        self.value = new_value

class Registers:
    def __init__(self):
        self.rList = [Register("r0",0),
                            Register("r1",0),
                            Register("r2",0),
                            Register("r3",0),
                            Register("r4",0),
                            Register("r5",0),
                            Register("r6",0),
                            Register("r7",0),
                            Register("r8",0),
                            Register("r9",0),
                            Register("r10",0),
                            Register("r11",0),
                            Register("r12",0),
                            Register("r13",0),
                            Register("r14",0),
                            Register("r15",0),
                            Register("r16",0),
                            Register("r17",0),
                             Register("r18",0),
                             Register("r19",0),
                             Register("r20",0),
                             Register("r21",0),
                             Register("r22",0),
                             Register("r23",0),
                             Register("r24",0),
                             Register("r25",0),
                             Register("r26",0),
                             Register("r27",0),
                             Register("r28",0),
                             Register("r29",0),
                             Register("r30",0),
                             Register("r31",0),
                             Register("$0A00",0),
                             Register("$0A30",0),
                             Register("$0A50",0),
                             Register("$0A80",0),
                             Register("0x0200",0),
                             Register("0x0201",0),
                             Register("0x0202",0),
                             Register("0x0203",0),
                             Register("0x3e",0),
                             Register("0x3f",0),
                             Register("0x3d",0),
                             # note: in operation on port 4 returns 68 for some reason
                             Register("4",68)]
        
        self.stack=[]
        # This is to store prior executions values when executing them.
        ## Note: the stored values is [current value in register1, current value in register 2] for each instruction.
        ## Note: current value referse to the values for the registers BEFORE executing the instruction. I.E. what is passed to it.
        #self.prior_Values=[-1,-1,-1,-1,-1,-1,-1,-1]
        
    def inRList(self, name):
        for i in range(len(self.rList)):
            if self.rList[i].getName() == name:
                return True
        else:
            return False
    
    def changeRegisterValue(self, name, value):
        for i in range(len(rList)):
            if self.rList[i].getName() == name:
                self.rList[i].changeValue(value)
                
    def printRegValues(self):
        for i in range(len(self.rList)):
            print("Name: " + self.rList[i].getName() + " Value: " + str(hex(self.rList[i].getValue()))[2:])
       
    def getLoc(self, name):
        for i in range(len(self.rList)):
            if self.rList[i].getName() == name:
                return i
            
    def getValue(self, command):
        value = self.rList[self.getLoc(command)].getValue()
        return value
    
    def changeValue(self, command, value):
        self.rList[self.getLoc(command)].changeValue(value)
    
    # performs the operation and stores the new value in the register.
    ## Note, for the Training Set files I programed them to only skip 
    ## one instruction when a branch, call, or jump occurs for simplicity.
    def performOp(self, Command):
        skip = 0
        
        ACommand = GetActualCommand(Command)
        current_Values = []
        new_Values = []
        maxnum = int('11111111',2)
        
        # All possible commands and operations
        if ACommand[0] == "add":
            ## get current
            v1 = self.getValue(ACommand[1]) 
            v2 = self.getValue(ACommand[2])
            current_Values = [v1, v2]
            ## perform Action
            ans = v1 + v2 
            if ans > maxnum:
                ans = ans - maxnum -1
            self.changeValue(ACommand[1], ans)
            ## get new values
            v1 = self.getValue(ACommand[1])
            v2 = self.getValue(ACommand[2])
            new_Values = [v1, v2]
            
        elif ACommand[0] == "adiw":
            # get current
            if ":" in ACommand[1]:
                r1, r2 = getRegs(ACommand[1])
                r1v = self.getValue(r1)
                r2v = self.getValue(r2)
                v1 = get2RegValue(r1v, r2v)
                v2 = getInt(ACommand[2])
                current_Values = [v1, v2]
                # perform Action
                nv1, nv2 = Opt2Regs(ACommand[0], r1v, r2v, int(ACommand[2]))
                self.changeValue(r1, nv1)
                self.changeValue(r2, nv2)
                ## get new values
                v1 = self.getValue(r1)
                v2 = self.getValue(r2)
                new_Values = [v1, v2]
            else:
                # get current
                v1 = self.getValue(ACommand[1])
                v2 = getInt(ACommand[2])
                current_Values = [v1]
                # perform Action
                ans = v1 + v2
                self.changeValue(ACommand[1], ans)
                ## get new values
                v1 = self.getValue(ACommand[1])
                new_Values = [v1]
            
        elif ACommand[0] == "andi":
            # get current
            v1 = self.getValue(ACommand[1])
            v2 = getInt(ACommand[2])
            current_Values = [v1, v2]
            # perform Action
            ans = v1 & v2
            self.changeValue(ACommand[1], ans)
            ## get new values
            v1 = self.getValue(ACommand[1])
            new_Values = [v1]
            
        elif ACommand[0] == "brne-t" or ACommand[0] == "breq-t":
            # get current
            # perform Action
            skip=1
            
        elif ACommand[0] == "brne-f" or ACommand[0] == "breq-f":
            # get current
            # perform Action
            skip=0
            
        elif ACommand[0] == "call":
            # get current
            # perform Action
            skip=0
        
        elif ACommand[0] == "clr":
            # get current
            v1 = self.getValue(ACommand[1])
            v2 = 0
            current_Values = [v1]
            # perform Action
            self.changeValue(ACommand[1], v2)
            ## get new values
            v1 = self.getValue(ACommand[1])
            new_Values = [v1]
            
        elif ACommand[0] == "eor":
            # get current
            v1 = self.getValue(ACommand[1])
            v2 = self.getValue(ACommand[2])
            current_Values = [v1, v2]
            # perform Action
            ans = v1 ^ v2
            self.changeValue(ACommand[1], ans)
            ## get new values
            v1 = self.getValue(ACommand[1])
            v2 = self.getValue(ACommand[2])
            new_Values = [v1, v2]
            
        elif ACommand[0] == "in":
            ## Note: the in command is hard to account for as the in takes
            ## input from another device. However for these experiments there is
            ## no direct other device. As such the in returns 0 most of the time.
            # Make sure to check that it does for the program and change if necessary.
            if self.inRList(ACommand[2]):
                v1 = self.getValue(ACommand[1])
                v2 = self.getValue(ACommand[2])
                current_Values = [v1, v2]
                # perform Action
                self.changeValue(ACommand[1], v2)
                ## get new values
                v1 = self.getValue(ACommand[1])
                new_Values = [v1, v2]
            else:
                v1 = self.getValue(ACommand[1])
                v2 = 0
                current_Values = [v1, v2]
                # perform Action
                self.changeValue(ACommand[1], v2)
                ## get new values
                v1 = self.getValue(ACommand[1])
                new_Values = [v1, v2]
                    
        elif ACommand[0] == "ldi":
            # get current
            v1 = self.getValue(ACommand[1])
            v2 = getInt(ACommand[2])
            current_Values = [v1, v2]
            # perform Action
            self.changeValue(ACommand[1], v2)
            ## get new values
            v1 = self.getValue(ACommand[1])
            new_Values = [v1]
            
        elif ACommand[0] == "lds":
            # get current
            v1 = self.getValue(ACommand[1])
            v2 = self.getValue(ACommand[2])
            current_Values = [v1, v2]
            # perform Action
            self.changeValue(ACommand[1], v2)
            ## get new values
            v1 = self.getValue(ACommand[1])
            v2 = self.getValue(ACommand[2])
            new_Values = [v1, v2]
            
        elif ACommand[0] == "or":
            # get current
            v1 = self.getValue(ACommand[1])
            v2 = self.getValue(ACommand[2])
            current_Values = [v1, v2]
            # perform Action
            ans = v1 | v2
            self.changeValue(ACommand[1], ans)
            ## get new values
            v1 = self.getValue(ACommand[1])
            v2 = self.getValue(ACommand[2])
            new_Values = [v1, v2]
            
        elif ACommand[0] == "out":
            ## Note: the in command is hard to account for as the in takes
            ## input from another device. However for these experiments there is
            ## no direct other device. As such the in returns 0 every time.
            # get current
            if self.inRList(ACommand[1]):
                v1 = self.getValue(ACommand[1])
                v2 = self.getValue(ACommand[2])
                current_Values = [v1, v2]
                # perform Action
                self.changeValue(ACommand[1], v2)
                ## get new values
                v1 = self.getValue(ACommand[1])
                new_Values = [v1, v2]
            else:
                print("ERROR: out register is not yet established.")
            
        elif ACommand[0] == "pop":
            # get current
            v1 = self.getValue(ACommand[1])
            current_Values = [v1]
            # perform Action
            ans = self.stack.pop()
            self.changeValue(ACommand[1], ans)
            ## get new values
            v1 = self.getValue(ACommand[1])
            new_Values = [v1]
            
        elif ACommand[0] == "push":
            # get current
            v1 = self.getValue(ACommand[1])
            current_Values = [v1]
            # perform Action
            ans = self.stack.append(v1)
          
        elif ACommand[0] == "rjmp":
            # get current
            # perform Action
            skip=1
            
        elif ACommand[0] == "sbiw":
            # get current
            if ":" in ACommand[1]:
                r1, r2 = getRegs(ACommand[1])
                r1v = self.getValue(r1)
                r2v = self.getValue(r2)
                v1 = get2RegValue(r1v, r2v)
                v2 = getInt(ACommand[2])
                current_Values = [v1, v2]
                # perform Action
                nv1, nv2 = Opt2Regs(ACommand[0], r1v, r2v, int(ACommand[2]))
                self.changeValue(r1, nv1)
                self.changeValue(r2, nv2)
                ## get new values
                v1 = self.getValue(r1)
                v2 = self.getValue(r2)
                new_Values = [v1, v2]
            else:
                # get current
                v1 = self.getValue(ACommand[1])
                v2 = getInt(ACommand[2])
                current_Values = [v1]
                # perform Action
                ans = v1 - v2
                self.changeValue(ACommand[1], ans)
                ## get new values
                v1 = self.getValue(ACommand[1])
                new_Values = [v1]
        
        elif ACommand[0] == "sub":
            # get current
            v1 = self.getValue(ACommand[1]) 
            v2 = self.getValue(ACommand[2])
            current_Values = [v1, v2]
            # perform Action
            ans = v1 - v2
            if ans < -1:
                ans = maxnum + ans +1
            self.changeValue(ACommand[1], ans)
            ## get new values
            v1 = self.getValue(ACommand[1])
            v2 = self.getValue(ACommand[2])
            new_Values = [v1, v2]
            
        elif ACommand[0] == "ret":
            # get current
            # perform Action
            skip =0
            
        elif ACommand[0] == "sts":
            # get current
            v1 = self.getValue(ACommand[1]) 
            v2 = self.getValue(ACommand[2])
            current_Values = [v1, v2]
            # perform Action
            self.changeValue(ACommand[1], v2)
            ## get new values
            v1 = self.getValue(ACommand[1])
            v2 = self.getValue(ACommand[2])
            new_Values = [v1, v2]
                    
        else:
            print("Unknown Command: " + ACommand[0])
            
            
            
        # Fill current values if no info if given.
        if len(current_Values) == 0:
            current_Values = [-1, -1]
            
        elif len(current_Values) == 1:
            current_Values.append(-1)
            
        if len(new_Values) == 0:
            new_Values = [-1, -1]
            
        elif len(new_Values) == 1:
            new_Values.append(-1)
            
        #print(ACommand[0])
            
        return ACommand[0], current_Values, new_Values, skip


# In[ ]:


def extractCodeAndValues(fileName):
    ASM_Insts = []
    current_Values = []
    new_Values = []
    
    file = open(fileName, "r")
    Lines = file.readlines()
    regValues = Registers()
    settingRegs = 0
    skip = 0
    first = 0
    for line in Lines:
        if skip == 0:
            words = line.split()
            # Cycle through the first instructions that sets the register values.
            if settingRegs == 0:
                if len(words) > 0:
                    if words[0] == "loop:":
                        settingRegs = 1
                    else:
                        if isInst(words[0]):
                            regValues.performOp(words)
            
            # perform the operations that do not set the registers.
            else:
                if first == 0:
                    first = 1
                    print("============= Before loop ==============")
                    regValues.printRegValues()
                    print("")
                if len(words) > 0:
                    if isInst(words[0]):
                        inst, c_values, n_values, skip = regValues.performOp(words)
                        current_Values.append(c_values)
                        new_Values.append(n_values)
                        ASM_Insts.append(inst) 
                            
        else:
            skip = skip-1             
                        
    print("============= After loop ==============")
    regValues.printRegValues()
    print("")
    return ASM_Insts, current_Values, new_Values


# In[ ]:


def andOpt(v1, v2):
    v1_bin = bin(v1)
    v2_bin = bin(v2)
    
    print(v1_bin[2:])
    print(v2_bin[2:])
    print(bin(v1 | v2)[2:])
    print(v1 | v2)


# In[ ]:


def check(ASM_Inst, c_Values, n_Values, loc=0):
    print("Instruction: " + ASM_Inst[loc])
    print("Current: ")
    print("1-dec: " + str(c_Values[loc][0]) + " 1-hex: " + hex(c_Values[loc][0])[2:])
    print("2-dec: " + str(c_Values[loc][1]) + " 2-hex: " + hex(c_Values[loc][1])[2:])
    print("New: ")
    print("1-dec: " + str(n_Values[loc][0]) + " 1-hex: " + hex(n_Values[loc][0])[2:])
    print("2-dec: " + str(n_Values[loc][1]) + " 2-hex: " + hex(n_Values[loc][1])[2:])


# In[ ]:


'''
Return a dictionary of 
(Data) datapoints of the major peak for the final instruction for the sequence.
(Code) the names of the instuctions being executed in the sequence. Includes the cycle at the end.
(Values) The value inside each register during each execution. -1 if not valid.
I.E. {"Data": [], "Code": [], "Values": []}

Note: Version two takes into account the instructions, 
    the values inside the registers during the start of the execution for each instructions,  
    and the cycle of the final peak.

Parameters
Signals: the signals to extract information from, Assumes that the provided info is a regular full signal.
ASM: The assembly instructions given in the same order as executed in the given signals.
Values: The values found during the start of the instruction being executed, given in the same order as in the given signal.
amount: size of the sequence to extract from.
pad: the average index distance between the top to the bottom of peaks. 
peakStart: the starting peak to start extracting information from. (This is because the start most likely dones have enough prior)
peakEnd: the ending peak to extract information from.
skip: amount of code to skip each time. Default = 1.
'''
def extractSequences_MajorPeak_V2(Signals, ASM, RValues, amount, pad, peakStart=-1, peakEnd=-1, skip=1):
    start_time = timeit.default_timer()
    
    # if -1 then assume to start when possible. I.E. start at the Amount signal.
    if peakStart == -1:
        startASM = ASM[:amount-1]
        
        peakStart = 0
        for inst in startASM:
            peakStart = peakStart + (Inst_Info.getCycles(inst)*2)
        
    #print(peakStart)
    
    # if -1 then assume the whole signal
    ## note yet implemented, will take to the end of signal every time 
    #if peakEnd == -1:
    #    peakEnd = getLength(Signals, pad)[0]
        
    #print(peakEnd)
    
    signals = MapTool.getPeaksDataset(Signals, pad)
    # list object to contain the dictionry information.
    listOfInfo = []
    peakLoc = 0
    # cycle through the code sequences to extract
    for c in range(0, ASM.shape[0]-(amount-1), skip):
        # obtain the ASM code for the sequence.
        ## list object to contain the exact code of the sequence.
        seqCode =[]
        ## cycle through the amount of code in the sequence
        for i in range(amount):
            seqCode.append(ASM[c+i])
          
        # obtain the register Values to extract.
        ## List object to contain the register values for the instructions inside the sequence.
        Values = []
        ## cycle through the amount of code in the sequence
        for i in range(amount):
            Values = Values + RValues[c+i]
        
        Values = np.array(Values)
        
        ## Convert to numpy
        seqCode = np.array(seqCode)
        
        for q in range(Inst_Info.getCycles(ASM[c+amount-1])):
            # add on the cycle information
            Code = np.copy(seqCode)
            Code= np.concatenate((Code,[q+1]))
            
            ## add the values
            Code= np.concatenate((Values, Code))
            
            ## Convert to numpy
            Code = np.array(Code)
            

            # Obtain the extracted majorpeaks from the signals.
            Data = []
            ## cycle through each signal
            for s in range(signals.shape[0]):
                # Note we need the second to last. last is the second peak in the cycle.
                # Also the peak amounts is from 0-total-1. So its -2
                majorpeak = [signals[s][(peakLoc*(2))+peakStart]]
                Data.append(majorpeak)

            Data = np.array(Data)
            
            
            peakLoc = peakLoc+1
            
            # Create a dictionary item pair of code and sequecnes.
            dictPair = {"Data": Data, "Code": Code}
            listOfInfo.append(dictPair)
    
    end_time = timeit.default_timer()
    
    print("Time: ", end_time-start_time)
    
    return listOfInfo


# In[ ]:


'''
This method will extract the sequences, assembly instruction, and register values of the sequences for multiple sets of signals.
It will also combine the like code sequences together.
This will get only the peaks

Note: Version two takes into account the instructions, 
    the values inside the registers during the start of the execution for each instructions,  
    and the cycle of the final peak.

Parameters
signalSets: the signal sets given in a 3D numpy matrix
Codes: full code of the extrancting sequences for the signals, given in a 2D numpy matrix
peakStarts: the starting peak for each sets EM signal to extract from, given in a 2D numpy matrix
peakEnds: the ending peak for each sets EM signal to extract from, given in a 2D numpy matrix
amount: the amount of instructions per sequence.
skip: the amount to of instructions to skip by per sequence extraction.
'''
def extractSequencesDataset_MajorPeak_V2(signalSets, ASMs, RValues, amount, pad, peakStarts=[], peakEnds=[], skip=1):
    # create a list of entire sequence information
    Info = []
    
    print("Extracting sequences...")
    # identify the sequences and sequence codes.
    ## Obtain the information for one set
    for n in range(ASMs.shape[0]):
        print("Working on Program: ", n)
        signals = signalSets[n]
        ASM = ASMs[n]
        RValue = RValues[n]
        peakStart = -1
        if len(peakStarts) != 0:
            peakStart = peakStart[n]
        peakEnd = -1
        if len(peakEnds) != 0:
            peakEnd = peakEnds[n]
        
        # obtain the sequence and code sequences information for the set
        if peakStart == -1 and peakEnd == -1:
            tempInfo = extractSequences_MajorPeak_V2(signals, ASM, RValue, amount, pad)
        else:
            tempInfo = extractSequences_MajorPeak_V2(signals, ASM, RValue, amount, pad, peakStart, peakEnd)
        
        # cycle through each obtained info
        for tempinfo in tempInfo:
            #print(tempinfo["Code"])
            # check if Info is empty
            if len(Info) == 0:
                # add the instances of the obtained info to Info
                dictSeq = {"Data": tempinfo["Data"], "Code": tempinfo["Code"]}
                Info.append(dictSeq)
                #print("Check")
            else:
                # check if there is already an occures of the sequence code.
                foundInst = False
                for info in Info:
                    if np.array_equal(info["Code"], tempinfo["Code"]):
                        # concatentated the two sets of sequences as they are for the same code sequence.
                        info["Data"] = np.concatenate((info["Data"], tempinfo["Data"]))
                        foundInst = True
                
                # if no already occures in Info list then add.
                if foundInst == False:
                    #print("Not Found")
                    # add the instances of the obtained info to Info
                    dictSeq = {"Data": tempinfo["Data"], "Code": tempinfo["Code"]}
                    Info.append(dictSeq)
                    
    return Info


# In[ ]:


'''
Converts the seq code information from assembly instruction into 
tokens from the given tokenizer.
'''
def convertASMToTokens_V2(Info, tokenizer, amount):
    Base = []
    for seq in Info:
        code = seq["Code"]
        #print(code)
        inst = code[amount*2:-1]
        for i in range(len(inst)):
            if "-t" in inst[i]:
                inst[i] = inst[i][:-2] + "T"
            elif "-f" in inst[i]:
                inst[i] = inst[i][:-2] + "F"
        
        #print(inst)
        tokens = tokenizer.texts_to_sequences(inst)
        token_list = []
        for token in tokens:
            if token == []:
                token_list.append(-1)
            else:
                token_list.append(token[0])
        new_code = np.copy(code[:amount*2].astype(np.int32))
        new_code = np.concatenate((new_code, token_list)) 
        new_code = np.append(new_code,code[amount*3].astype(np.int32))
        
        Base.append(new_code)
    return np.array(Base)


# In[ ]:


def convertToGANsTrainSet_Balance_V2(Info, tokenizer, max_seq, amount, random_seed=0):
    start = timeit.default_timer()
    
    count = 1
    first = True
    for info in Info:
        seqs = np.copy(info["Data"])
        np.random.seed(random_seed)
        random.shuffle(seqs)
        seqs = seqs[:amount]
        
        code = info["Code"].tolist()
        inst = code[max_seq*2:-1]
        for i in range(len(inst)):
            if "-t" in inst[i]:
                inst[i] = inst[i][:-2] + "T"
            elif "-f" in inst[i]:
                inst[i] = inst[i][:-2] + "F"
        
        tokens = tokenizer.texts_to_sequences(inst)
        token_list = []
        for token in tokens:
            if token == []:
                token_list.append(-1)
            else:
                token_list.append(token[0])
                
        #print(code)
        new_code = np.copy(code[:max_seq*2])
        new_code = np.concatenate((new_code, token_list)) 
        new_code = np.append(new_code,code[max_seq*3])
        new_code = np.array(new_code).astype(np.int32)
        
        # to be implemented later.
        # full with null value to indicate cut off if code sequence is less thatn max_seq
        #is_correct_length = False
        #while is_correct_length == False:
        #    if len(code_tok[0]) == max_seq:
        #        is_correct_length = True
        #    else:
        #        null_code = [-1]
        #        code_tok[0] = null_code + code_tok[0]
        #code_tok = np.array(code_tok)
        
        codes = np.tile(new_code, (seqs.shape[0],1))
        
        if first:
            all_seq = np.array(seqs)
            all_code = np.array(codes)
            first = False
        
        else:
            all_seq = np.concatenate((all_seq, seqs))
            all_code = np.concatenate((all_code, codes))
            
        if count %500 == 0:
            print(count)
        count = count + 1
    
    all_seq = np.array(all_seq)
    all_code = np.array(all_code)
    
    stop = timeit.default_timer()
    print('Time: ' + str(stop - start))
    
    return all_seq, all_code


# In[ ]:


def convertToGANsTrainSet_V2(Info, tokenizer, max_seq, amount, random_seed=0):
    start = timeit.default_timer()
    
    first = True
    for info in Info:
        seqs = np.copy(info["Data"])
        np.random.seed(random_seed)
        random.shuffle(seqs)
        
        code = info["Code"].tolist()
        inst = code[max_seq*2:-1]
        for i in range(len(inst)):
            if "-t" in inst[i]:
                inst[i] = inst[i][:-2] + "T"
            elif "-f" in inst[i]:
                inst[i] = inst[i][:-2] + "F"
        
        tokens = tokenizer.texts_to_sequences(inst)
        token_list = []
        for token in tokens:
            if token == []:
                token_list.append(-1)
            else:
                token_list.append(token[0])
                
        #print(code)
        new_code = np.copy(code[:max_seq*2])
        new_code = np.concatenate((new_code, token_list)) 
        new_code = np.append(new_code,code[max_seq*3])
        #print(new_code)
        
        # to be implemented later.
        # full with null value to indicate cut off if code sequence is less thatn max_seq
        #is_correct_length = False
        #while is_correct_length == False:
        #    if len(code_tok[0]) == max_seq:
        #        is_correct_length = True
        #    else:
        #        null_code = [-1]
        #        code_tok[0] = null_code + code_tok[0]
        #code_tok = np.array(code_tok)
        
        codes = np.tile(new_code, (seqs.shape[0],1))
        
        if first:
            all_seq = np.array(seqs)
            all_code = np.array(codes)
            first = False
        
        else:
            all_seq = np.concatenate((all_seq, seqs))
            all_code = np.concatenate((all_code, codes))
    
    all_seq = np.array(all_seq)
    all_code = np.array(all_code)
    
    stop = timeit.default_timer()
    print('Time: ' + str(stop - start))
    
    return all_seq, all_code

