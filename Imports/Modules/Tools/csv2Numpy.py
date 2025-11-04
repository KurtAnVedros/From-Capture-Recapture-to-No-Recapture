#!/usr/bin/env python
# coding: utf-8

# # Code is for converting csvs to numpy arrays

# In[8]:


from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
import timeit
import csv
import os
import math 
import random as rand
from numpy import genfromtxt

import warnings


# In[9]:


params = {'legend.fontsize': 50,
          'figure.figsize': (45, 15),
         'axes.labelsize': 60,
         'axes.titlesize':70,
         'xtick.labelsize':50,
         'ytick.labelsize':50,
         'lines.linewidth': 5}
plt.rcParams.update(params)


# # Methods

# In[10]:


'''
Methods defines where to "Cut" the signal based on the threshold. I.E. above a point of the b channel signal.

array: the numpy signal array.
skipStart: dones not account for the first someodd ofthe signal when looking for points above threshold.
thresh: the threshold.
direction: wether to look above or below thresh.
'''
def findCuttingPoint(array, skipStart, thresh, direction= 0):
    stop = 0
    cuttingpoint = 0
    index = skipStart
    while stop == 0:
        index = index + 1
        if direction == 0:
            if array[index] < thresh:
                cuttingpoint = index
                stop = 1
        else:
            if array[index] > thresh:
                cuttingpoint = index
                stop = 1
            
    return cuttingpoint


# In[11]:


'''
Gets the best cutting point found based on the information given. 

Folder: Folder location of the csv files
folderDepth: how deep you need to go in the folder section to get to the  csv files.
skipStart: dones not account for the first someodd ofthe signal when looking for points above threshold.
thresh: the threshold.
'''

def getCutLocation(Folder, folderDepth=1, skipStart=100, thresh=0, direction= 0):
    CuttingPoint_list = []
    folderName = str(Folder) +"/"
    folder = os.listdir(Folder)
    if folderDepth == 1:
        for file in folder:
            try:      
                loadFileName = folderName + str(file)
                TempSignal_3sections = genfromtxt(loadFileName, delimiter=',')
                TempSignal_B = TempSignal_3sections[2:TempSignal_3sections.shape[0],2]

                CuttingPoint_list.append(findCuttingPoint(TempSignal_B,skipStart, thresh))
            except FileNotFoundError as error:
                print(error)
    
    elif folderDepth == 2:
        for Folder_1 in folder:
            folder_1 = os.listdir(Folder + Folder_1)
            folder_1Name = Folder_1 + "/"
            for file in folder_1:
                try:      
                    loadFileName = folderName + folder_1Name + str(file)
                    TempSignal_3sections = genfromtxt(loadFileName, delimiter=',')
                    TempSignal_B = TempSignal_3sections[2:TempSignal_3sections.shape[0],2]

                    CuttingPoint_list.append(findCuttingPoint(TempSignal_B,skipStart, thresh, direction))
                except FileNotFoundError as error:
                    print(error)
            
    FinalList = list(dict.fromkeys(CuttingPoint_list))
    print("Possible cuttinging points found for individual", FinalList)
    BestCP = max(FinalList)
    return BestCP


# In[ ]:


'''
Gets the best cutting point found based on the information given. 

Folder: Folder location of the csv files
folderDepth: how deep you need to go in the folder section to get to the  csv files.
skipStart: dones not account for the first someodd ofthe signal when looking for points above threshold.
thresh: the threshold.
'''

def getCutLocationCount(Folder, folderDepth=1, skipStart=100, thresh=0, direction= 0, amount = 100):
    CuttingPoint_list = []
    folderName = str(Folder) +"/"
    folder = os.listdir(Folder)
    count = 0
    if folderDepth == 1:
        for file in folder:
            if count < amount:
                try:      
                    loadFileName = folderName + str(file)
                    TempSignal_3sections = genfromtxt(loadFileName, delimiter=',')
                    TempSignal_B = TempSignal_3sections[2:TempSignal_3sections.shape[0],2]

                    CuttingPoint_list.append(findCuttingPoint(TempSignal_B,skipStart, thresh))
                except FileNotFoundError as error:
                    print(error)
                count = count + 1
    
    elif folderDepth == 2:
        for Folder_1 in folder:
            folder_1 = os.listdir(Folder + Folder_1)
            folder_1Name = Folder_1 + "/"
            for file in folder_1:
                if count < amount:
                    try:      
                        loadFileName = folderName + folder_1Name + str(file)
                        TempSignal_3sections = genfromtxt(loadFileName, delimiter=',')
                        TempSignal_B = TempSignal_3sections[2:TempSignal_3sections.shape[0],2]

                        CuttingPoint_list.append(findCuttingPoint(TempSignal_B,skipStart, thresh, direction))
                    except FileNotFoundError as error:
                        print(error)
                    count = count + 1
            
    FinalList = list(dict.fromkeys(CuttingPoint_list))
    print("Possible cuttinging points found for individual", FinalList)
    BestCP = max(FinalList)
    return BestCP


# In[12]:


'''
Process of loading and cutting all csv files that contain the signal with a and b channel information.

Folder: Folder location of the csv files
folderDepth: how deep you need to go in the folder section to get to the  csv files.
skipStart: dones not account for the first someodd ofthe signal when looking for points above threshold.
thresh: the threshold.
toCut: if to cut the signal given the b channel
'''
def csv2numpy(Folder, folderDepth=1, skipStart=100, thresh=0, toCut = True, direction= 0):
    start = timeit.default_timer()
    if toCut:
        CP = getCutLocation(Folder, folderDepth, skipStart, thresh, direction)
        signals_list = []
        folderName = str(Folder) +"/"
        folder = os.listdir(Folder)
        if folderDepth == 1:
            for file in folder:
                try:
                    loadFileName = folderName + str(file)
                    TempSignal_3sections = genfromtxt(loadFileName, delimiter=',')
                    TempSignal_A = TempSignal_3sections[2:TempSignal_3sections.shape[0],1]
                    Signal = TempSignal_A[:CP]
                    signals_list.append(Signal)
                except FileNotFoundError as error:
                    print(error)
        elif folderDepth == 2:
            for Folder_1 in folder:
                folder_1 = os.listdir(Folder + Folder_1)
                folder_1Name = Folder_1 + "/"
                for file in folder_1:
                    try:
                        loadFileName = folderName + folder_1Name + str(file)
                        TempSignal_3sections = genfromtxt(loadFileName, delimiter=',')
                        TempSignal_A = TempSignal_3sections[2:TempSignal_3sections.shape[0],1]
                        Signal = TempSignal_A[:CP]
                        signals_list.append(Signal)
                    except FileNotFoundError as error:
                        print(error)

        signals = np.array(signals_list, dtype=object) 
        stop = timeit.default_timer()
        
    else:
        signals_list = []
        folderName = str(Folder) +"/"
        folder = os.listdir(Folder)
        if folderDepth == 1:
            for file in folder:
                try:
                    loadFileName = folderName + str(file)
                    TempSignal_3sections = genfromtxt(loadFileName, delimiter=',')
                    TempSignal_A = TempSignal_3sections[2:TempSignal_3sections.shape[0],1]
                    Signal = TempSignal_A
                    signals_list.append(Signal)
                except FileNotFoundError as error:
                    print(error)
        elif folderDepth == 2:
            for Folder_1 in folder:
                folder_1 = os.listdir(Folder + Folder_1)
                folder_1Name = Folder_1 + "/"
                for file in folder_1:
                    try:
                        loadFileName = folderName + folder_1Name + str(file)
                        TempSignal_3sections = genfromtxt(loadFileName, delimiter=',')
                        TempSignal_A = TempSignal_3sections[2:TempSignal_3sections.shape[0],1]
                        Signal = TempSignal_A
                        signals_list.append(Signal)
                    except FileNotFoundError as error:
                        print(error)

        signals = np.array(signals_list, dtype=object) 
        stop = timeit.default_timer()
    
    print('Time: ', stop - start)  
    
    return signals


# In[ ]:


'''
Process of loading and cutting all csv files that contain the signal with a and b channel information.

Folder: Folder location of the csv files
folderDepth: how deep you need to go in the folder section to get to the  csv files.
skipStart: dones not account for the first someodd ofthe signal when looking for points above threshold.
thresh: the threshold.
toCut: if to cut the signal given the b channel
'''
def csv2numpyCount(Folder, folderDepth=1, skipStart=100, thresh=0, toCut = True, direction= 0, amount = 100):
    start = timeit.default_timer()
    if toCut:
        CP = getCutLocationCount(Folder, folderDepth, skipStart, thresh, direction, amount=amount)
        signals_list = []
        folderName = str(Folder) +"/"
        folder = os.listdir(Folder)
        count = 0
        if folderDepth == 1:
            for file in folder:
                if count < amount:
                    try:
                        loadFileName = folderName + str(file)
                        TempSignal_3sections = genfromtxt(loadFileName, delimiter=',')
                        TempSignal_A = TempSignal_3sections[2:TempSignal_3sections.shape[0],1]
                        Signal = TempSignal_A[:CP]
                        signals_list.append(Signal)
                    except FileNotFoundError as error:
                        print(error)
                    count = count + 1
        elif folderDepth == 2:
            for Folder_1 in folder:
                folder_1 = os.listdir(Folder + Folder_1)
                folder_1Name = Folder_1 + "/"
                for file in folder_1:
                    if count < amount:
                        try:
                            loadFileName = folderName + folder_1Name + str(file)
                            TempSignal_3sections = genfromtxt(loadFileName, delimiter=',')
                            TempSignal_A = TempSignal_3sections[2:TempSignal_3sections.shape[0],1]
                            Signal = TempSignal_A[:CP]
                            signals_list.append(Signal)
                        except FileNotFoundError as error:
                            print(error)
                        count = count + 1

        signals = np.array(signals_list, dtype=object) 
        stop = timeit.default_timer()
        
    else:
        signals_list = []
        folderName = str(Folder) +"/"
        folder = os.listdir(Folder)
        count = 0
        if folderDepth == 1:
            for file in folder:
                if count < amount:
                    try:
                        loadFileName = folderName + str(file)
                        TempSignal_3sections = genfromtxt(loadFileName, delimiter=',')
                        TempSignal_A = TempSignal_3sections[2:TempSignal_3sections.shape[0],1]
                        Signal = TempSignal_A
                        signals_list.append(Signal)
                    except FileNotFoundError as error:
                        print(error)
                    count = count +1
        elif folderDepth == 2:
            for Folder_1 in folder:
                folder_1 = os.listdir(Folder + Folder_1)
                folder_1Name = Folder_1 + "/"
                for file in folder_1:
                    if count < amount:
                        try:
                            loadFileName = folderName + folder_1Name + str(file)
                            TempSignal_3sections = genfromtxt(loadFileName, delimiter=',')
                            TempSignal_A = TempSignal_3sections[2:TempSignal_3sections.shape[0],1]
                            Signal = TempSignal_A
                            signals_list.append(Signal)
                        except FileNotFoundError as error:
                            print(error)
                    count = count +1
                

        signals = np.array(signals_list, dtype=object) 
        stop = timeit.default_timer()
    
    print('Time: ', stop - start)  
    
    return signals


# In[13]:


'''
Removes any nan or inf sample points in the dataset and changes them to 0.

Dataset: the dataset.
'''
def removeNanAndInf(dataset):
    remake_Dataset = np.copy(dataset)
    for signal_Index in range(dataset.shape[0]):
        for sample_Index in range(dataset.shape[1]):
            if np.isnan(dataset[signal_Index][sample_Index]):
                remake_Dataset[signal_Index][sample_Index] = 0
            if np.isinf(dataset[signal_Index][sample_Index]):
                remake_Dataset[signal_Index][sample_Index] = 0
    return remake_Dataset


# In[17]:


'''
Gets a sample to show what the inputs will do to all signals

Folder: Folder location of the csv files
skipStart: dones not account for the first someodd ofthe signal when looking for points above threshold.
thresh: the threshold.
toCut: if to cut the signal given the b channel
'''

def sample(Folder, folderDepth=1, skipStart=100, thresh=0, toCut = True, B_channel = True, direction= 0):

    plt.rcParams.update(params)
    
    if toCut:
        CP = getCutLocation(Folder, folderDepth, skipStart, thresh, direction)
        signals_list = []
        folderName = str(Folder) +"/"
        folder = os.listdir(Folder)

        file = folder[0]
        if folderDepth == 2:
            folder = os.listdir(Folder + file)
            folderName = str(Folder) + str(file) + "/"
            print(folderName)
            file = folder[0]
        try:
            loadFileName = folderName + str(file)
            TempSignal_3sections = genfromtxt(loadFileName, delimiter=',')
            TempSignal_A = TempSignal_3sections[2:TempSignal_3sections.shape[0],1]
            TempSignal_B = TempSignal_3sections[2:TempSignal_3sections.shape[0],2]
            time = np.arange(TempSignal_A.shape[0])

            plt.plot(time, TempSignal_A, label="A-channel")
            plt.plot(time, TempSignal_B, label="B-channel")
            plt.title("Before Cutting.")
            plt.legend()
            plt.show()
            
            plt.plot(time, TempSignal_B)
            plt.title("B-channel.")
            plt.show()


            Signal = TempSignal_A[:CP]

            time = np.arange(Signal.shape[0])
            plt.plot(time, Signal)
            plt.title("After Cutting")
            plt.show()

        except FileNotFoundError as error:
            print(error)
            
    else:
        signals_list = []
        folderName = str(Folder) +"/"
        folder = os.listdir(Folder)

        file = folder[0]
        if folderDepth == 2:
            folder = os.listdir(Folder + file)
            folderName = str(Folder) + str(file) + "/"
            print(folderName)
            file = folder[0]
        try:
            if B_channel:
                loadFileName = folderName + str(file)
                TempSignal_3sections = genfromtxt(loadFileName, delimiter=',')
                TempSignal_A = TempSignal_3sections[2:TempSignal_3sections.shape[0],1]
                TempSignal_B = TempSignal_3sections[2:TempSignal_3sections.shape[0],2]
                time = np.arange(TempSignal_A.shape[0])

                plt.plot(time, TempSignal_A)
                plt.plot(time, TempSignal_B)
                plt.title("Before cutting.")
                plt.show()
            else:
                loadFileName = folderName + str(file)
                TempSignal_3sections = genfromtxt(loadFileName, delimiter=',')
                TempSignal_A = TempSignal_3sections[2:TempSignal_3sections.shape[0],1]
                #TempSignal_B = TempSignal_3sections[2:TempSignal_3sections.shape[0],2]
                time = np.arange(TempSignal_A.shape[0])

                plt.plot(time, TempSignal_A)
                #plt.plot(time, TempSignal_B)
                plt.title("Before cutting.")
                plt.show()


            Signal = TempSignal_A

            time = np.arange(Signal.shape[0])
            plt.plot(time, Signal)
            plt.title("After cutting")
            plt.show()

        except FileNotFoundError as error:
            print(error)


# In[ ]:


'''
Gets a sample to show what the inputs will do to all signals

Folder: Folder location of the csv files
skipStart: dones not account for the first someodd ofthe signal when looking for points above threshold.
thresh: the threshold.
toCut: if to cut the signal given the b channel
'''

def getSampleInfo(Folder, folderDepth=1, skipStart=100, thresh=0, toCut = True, B_channel = True, direction= 0):

    plt.rcParams.update(params)
    
    if toCut:
        CP = getCutLocation(Folder, folderDepth, skipStart, thresh, direction)
        signals_list = []
        folderName = str(Folder) +"/"
        folder = os.listdir(Folder)

        file = folder[0]
        if folderDepth == 2:
            folder = os.listdir(Folder + file)
            folderName = str(Folder) + str(file) + "/"
            print(folderName)
            file = folder[0]
        try:
            loadFileName = folderName + str(file)
            TempSignal_3sections = genfromtxt(loadFileName, delimiter=',')
            TempSignal_A = TempSignal_3sections[2:TempSignal_3sections.shape[0],1]
            TempSignal_B = TempSignal_3sections[2:TempSignal_3sections.shape[0],2]
            time = np.arange(TempSignal_A.shape[0])

            plt.plot(time, TempSignal_A)
            plt.plot(time, TempSignal_B)
            plt.title("Before cutting.")
            plt.show()
            
            plt.plot(time, TempSignal_B)
            plt.title("B-channel.")
            plt.show()
            
            Signal_A = TempSignal_A
            Signal_B = TempSignal_B


            Signal = TempSignal_A[:CP]

            time = np.arange(Signal.shape[0])
            plt.plot(time, Signal)
            plt.title("After cutting")
            plt.show()
            
            Signal_ACut = TempSignal_A[:CP]
            Signal_BCut = TempSignal_B[:CP]
            

        except FileNotFoundError as error:
            print(error)
        
        return Signal_A, Signal_B, Signal_ACut, Signal_BCut
            
    else:
        return 0, 0, 0, 0

