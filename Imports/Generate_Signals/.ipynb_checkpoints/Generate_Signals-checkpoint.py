#!/usr/bin/env python
# coding: utf-8

# This file contains methods for generating signals

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
import string

#import torch
#from torch import nn

import warnings
from scipy.spatial import distance


from importlib import reload
import sys
sys.path.append("../Imports/") # Adds higher directory to python modules path.

from Modules.Tools import MapTool
from Modules.Tools import BoxPlot
from Modules.Tools import Extractor
from Modules.Tools import peakCorrelation as pC
from Modules.Tools import ATmega2560_Instructions as Inst_Info


# In[ ]:


'''
Removes the prior and after information by having a start and end to the mid point.
'''
def removePriorandAfterInfo(cycle, midpoint=0):
    start = 0
    end = len(cycle)-1
    
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
    
    return cycle[start:end]


# In[ ]:


'''
This method will take EM signals' major peaks and stitch them together in the order given returning 1 EM signal of major peaks only.

dataset: a 2D numpy array of the generated EM signals to be stitched together.
amount: amount of instruction contained in the EM signal
pad: The padding amount that referse to the amount of time index it takes to complete half a peak to peak interval
midpoint: the common mid or avg point between the high and low peaks.
'''
def EMstitching(dataset, pad, midpoint=0):
    # list of EMs to stitch together.
    first = True
    # cycle through the dataset
    for n in range(dataset.shape[0]):
        cycle = dataset[n]
        
        # obtain the EM of the important instruction from generated EM sequence of instructions
        ## this removes some of the start and end to obtain the important area.
        cycle = removePriorandAfterInfo(cycle, midpoint)
        
        # stitch together into one EM
        if first == True:
            EMFinal = np.copy(cycle)
            first = False
        else:
            EMFinal = np.concatenate((EMFinal, cycle))
    
    EMFinal = np.array(EMFinal)
    return EMFinal


# In[ ]:


'''
This method will take EM signals' major peaks and stitch them together in the order given returning 1 EM signal of major peaks only.

dataset: a 3D numpy array of the generated EM signals to be stitched together.
amount: amount of instruction contained in the EM signal
pad: The padding amount that referse to the amount of time index it takes to complete half a peak to peak interval
midpoint: the common mid or avg point between the high and low peaks.
'''
def EMstitching_Dataset(dataset, pad, midpoint=0):
    new_dataset = []
    
    # list of EMs to stitch together.
    first = True
    # cycle through the dataset
    for i in range(dataset.shape[1]):
        new_dataset.append(EMstitching(dataset[:,i], pad, midpoint=0))         
    
    return new_dataset

# In[ ]:


'''
Given lists of ASM_Inst, c_Values,  states of each Training Set in a list and a list of numpy arrays of sequence signals,

Input:
    train_Set_Extracted: list of ASM_Inst, list of c_Values, list of states of each Train_Program in that order.
    train_Set_Signals: list of numpy arrays of sequence signals. (must be in the same order as Train_Set_Extracted)
    sequence_size: sequence size of the number of instructions (prior instructions only) to consider.

Output:
    Sequence_Code: list of sequence code for each program executed in order of given Training_Set_Extracted.
    Register_Values: list of register values for each instruction in the each sequence for each program executed in order of give Training_Set_Extracted.
    Target_States: list of the state prior to executing the target instruction.
    Signal_Priors: list of signals corresponding to the prior instructions. 
        (Note: will be prior + 1 = same as sequence_size as Transformers need the prior signal to generate a new signal of same shape.
            This new signal will include some of the prior signal with added new generated time index values/amplitudes)
'''
def extractInfoFromTrainingSets_ToGen(train_Set, train_Code, sequence_size, tokenizer,
                                       include_c_Values=True, include_States=True, pad = 7, set_length = 35, amount = 3000, 
                                       to_Tokenize=True):
    x_repeats = []
    start_peak = -2
    end_peak = 1
    
    # First obtain the Sequence_Code, Register_Values and Target_States
    # Adjust how much to skip at the start for each cycle that is a prior instruction for first target instruction.
    for i in range(sequence_size):
        prior_inst = train_Code[i]
        amount_Cycles = Inst_Info.getCycles(prior_inst)
        for n in range(amount_Cycles):
            start_peak = start_peak + 2
            end_peak = end_peak +2

    ## Obtain the assembly instructions (ASM_Inst), current values (c_Values), and states.
    ASM_Inst, c_Values, states = train_Set

    ## Cycle though each instance
    for i in range(len(ASM_Inst)-sequence_size):
        info=[]
        target = ""
        ## Cycle through sequence_size
        for n in range(sequence_size+1):
            ## obtain the instruction in the sequence
            if to_Tokenize:
                ASM = [ASM_Inst[i+n]][0]
                if "-t" in ASM:
                    ASM = ASM[:-2] + "T"
                elif "-f" in ASM:
                    ASM = ASM[:-2] + "F"

                info.append(tokenizer.texts_to_sequences([ASM])[0][0])
            else:
                info.append(ASM_Inst[i+n])
            target = ASM_Inst[i+n]

            ## obtain current values before executing each instruction in the sequence
            if include_c_Values:
                info.append(c_Values[i+n][0])
                info.append(c_Values[i+n][1])

        ## Add the state before executing the target instruction
        if include_States:
            info = info + states[i+sequence_size]

        ## Since the model generates cycles (2 waves) at a time, we need to check how many cycles the target instruction
        ## has and also add the info for each cycle along with an indicator on which cycle to generate.
        #print(target)
        #print(Inst_Info.getCycles(target))
        amount_Cycles = Inst_Info.getCycles(target)
        for i in range(amount_Cycles):
            info_cycle = info.copy()
            info_cycle.append(i+1)
            x_repeats.append(info_cycle)

    print(len(x_repeats))
            
    return x_repeats