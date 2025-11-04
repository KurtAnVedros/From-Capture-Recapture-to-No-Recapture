#!/usr/bin/env python
# coding: utf-8

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

    #print(len(x_repeats))
            
    return x_repeats


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
def extractInformationFromTrainingSets(train_Set_Extracted, train_Set_Codes, sequence_size, tokenizer,
                                       include_c_Values=True, include_States=True, pad = 7, set_length = 35, amount = 3000, 
                                       to_Tokenize=True):
    x_repeats = []
    y_repeats = []
    start_peak = -2
    end_peak = 1
    
    # First obtain the Sequence_Code, Register_Values and Target_States
    ## cycle through each training set
    for t in range(train_Set_Extracted.shape[0]):
        train_Set = train_Set_Extracted[t]
        train_Code = train_Set_Codes[t]
        
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


# In[ ]:


def Convert2BalanceTrainDataset(x, y, amount=3000, random_seed = 20):
    random.seed(random_seed)
    
    new_X = [x[0].tolist()]
    new_Y = [y[0].tolist()]
    
    first = 0
    # Remove repeatative x inputs.
    for i in range(x.shape[0]-2):
        if x[i+1].tolist() in new_X:
            #print("X input seen before: ", x[i+1])
            index = new_X.index(x[i+1].tolist())
            new_Y[index] = new_Y[index] + y[i+1].tolist()
            #print(x[i+1].tolist())
            #print(len(new_Y[index]))
            
        else:
            new_X.append(x[i+1].tolist())
            new_Y.append(y[i+1].tolist())
            
    # Shuffle all that is past the set amount and take only the set amount.
    ## This randomly takes samples of the given x_input
    for i in range(len(new_Y)):
        #random.shuffle(new_Y[i])
        if len(new_Y[i]) > amount:
            new_Y[i] = new_Y[i][:amount]
            
    
    # Now we need to tile and have a x for every y.
    ## tile x
    unique_instance = len(new_X)
    new_X = np.tile(new_X, (2500, 1))
    
    new_new_Y = []
    for i in range(2500):
        for a_list in new_Y:
            new_new_Y.append(a_list[i])
            
    del new_Y
            
    return np.array(new_X), np.array(new_new_Y, dtype=object), unique_instance

