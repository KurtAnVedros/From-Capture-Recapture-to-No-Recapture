#!/usr/bin/env python
# coding: utf-8

# # Notebook to illustrate the Synth Data Generator function.
# - Function will have the following functions.
#     - Generates synth data that looks close to ElectoMagnetic signal from a PLC.
#     - Able to generate a Normal and Anomalous datasets of various types of Amplidue, Polutions Rate, Polution Location, Sampling Rate, and SNR
#     - Generates anomalys that is the same based on anomaly random seed, as to indicate the same anomalous function being run.
#     
#     
# - Following perameters are given
#     - Number of cycles.
#         - A cycle is considered as one instruction and is two pecks.
#     - Sample Rate.
#         - The amount of samples taken per peak.
#             - 2 = two per peak which is 2x
#     - Amount of Signals to generate
#     - Amplidue Range
#     - Fluctuation Range
#         - Should be very small in comparison to amplidue.
#     - Pollution Rate
#         - Amount of anomaly instruction is present.
#     - Pollution Location
#         - Location where the pollution starts, Should be given as percentage, I.E. 50 = %50
#     - Signal To Noise Ratio, SNR.
#         - Given in dbm
#     - Normal Random Seed. 
#         - (Used as to get the same output of Normal Signal from the function every time if needed.)
#     - Anomaly Random Seed. 
#         - (Used as to get the same output of Anomaly Signal from the function every time if needed. Should be different from the Normal Signal at the insertion point.)

# In[2]:


import numpy as np
from matplotlib import pyplot as plt
import math 
import random as rand
from numpy import sum,isrealobj,sqrt
from numpy.random import standard_normal

# Changes size of plots
plt.rcParams['figure.figsize'] = [15, 5]


# In[3]:


# Graphs a signal
def graph(signal, time, title, limY = 0):
    plt.plot(time, signal)
    plt.title(title)
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    if limY > 0:
        plt.ylim(-limY, limY)
    plt.show()


# ## Generate a Standard Signal

# In[4]:


def generateSignal(instructions, sampleRate):
    sample_rate = sampleRate

    start_time = 0
    end_time = instructions * 2

    time = np.arange(start_time, end_time, 1/sample_rate)
    timeLength = range(0,sample_rate*10)

    frequency = 1
    amplitude = 1

    theta = 0

    signal = amplitude * np.sin(2 * np.pi * frequency * time + theta)
    
    return signal, time


# ## Alter signal to change amplitudes

# In[8]:


def addAmplidue(signal, ampRange, instructions, sampleRate, randomSeed):
    rand.seed(randomSeed)
    randAmp = np.zeros(instructions*2*sampleRate)
    randInt2 = 0

    for index in range(instructions):
        # first peak in cycle
        randInt1 = ampRange[rand.randint(0,ampRange.shape[0]-1)]
        check = 0
        while check == 0:
            if randInt1 == randInt2:
                randInt1 = ampRange[rand.randint(0,ampRange.shape[0]-1)]
            else:
                check = 1
        for times in range(sampleRate):
            randAmp[index*(sampleRate*2)+times] = randInt1
        # second peack in cycle
        randInt2 = ampRange[rand.randint(0,ampRange.shape[0]-1)]
        for times in range(sampleRate):
            randAmp[index*(sampleRate*2)+times+sampleRate] = randInt2

    ampSignal = signal * randAmp
    return ampSignal


# ## Alter signal to add fluctuation (normal to see slight variation between signal readings)

# In[10]:


def addFluctuation(signal, flucRange):
    flucSignal = np.copy(signal)
    for index in range(signal.shape[0]):
        # first peak in cycle
        randFloat = rand.uniform(-flucRange, flucRange)
        flucSignal[index] =  flucSignal[index] + randFloat

    return flucSignal


# ## Alters the signal to add an anomaly to it.

# In[12]:


def addPollution(normalSignal, instructions, sampleRate, ampRange, pollutionRate, pollutionLoc, randomSeed):
    anomalySignal, timeTemp = generateSignal(instructions, sampleRate)
    anomalySignal = addAmplidue(anomalySignal, ampRange, instructions, sampleRate, randomSeed)
    
    pollinstructionstart = int(instructions*2*(pollutionLoc/100))
    pollRateinstructions = int(instructions*2*(pollutionRate/100))
    
    # check to make sure that the anomaly signal is different then the normal signal.
    check = 0
    count = 1
    while check == 0:
        # check if same signal as normal
        if np.array_equal(normalSignal, anomalySignal):
            # change to new anomaly signal instruction
            anomalySignal, timeTemp = generateSignal(instructions, sampleRate)
            anomalySignal = addAmplidue(anomalySignal, ampRange, instructions, sampleRate, randomSeed+count)
            count = 1
        # check if first anomaly instruction is same as normal instruction it is replacing.
        elif normalSignal[pollinstructionstart*sampleRate] == anomalySignal[0]:
            # change to new anomaly signal instruction
            anomalySignal, timeTemp = generateSignal(instructions, sampleRate)
            anomalySignal = addAmplidue(anomalySignal, ampRange, instructions, sampleRate, randomSeed+count)
            count = 1
        else:
            check = 1
    
    anomaly_1Sec = normalSignal[0:pollinstructionstart*sampleRate]
    anomaly_2Sec = anomalySignal[0:pollRateinstructions*sampleRate]
    anomaly_3Sec = normalSignal[pollinstructionstart*sampleRate: (normalSignal.shape[0]- (pollRateinstructions*sampleRate))]
    
    anomaly = np.concatenate((anomaly_1Sec, anomaly_2Sec ,anomaly_3Sec))
    
    return anomaly


# ## Adds noise to the anomaly

# In[22]:


# author - Mathuranathan Viswanathan (gaussianwaves.com
# This code is part of the book Digital Modulations using Python

def addNoise(s,SNRdB,L=1):   
    gamma = 10**(SNRdB/10)             # SNR to linear scale
    if s.ndim==1:                      # if s is single dimensional vector
        P=L*sum(abs(s)**2)/len(s)      # Actual power in the vector
    else:                              # multi-dimensional signals like MFSK
        P=L*sum(sum(abs(s)**2))/len(s)     # if s is a matrix [MxN]
    N0=P/gamma                         # Find the noise spectral density
    if isrealobj(s):                   # check if input is real/complex object type
        n = sqrt(N0/2)*standard_normal(s.shape) # computed noise
    else:
        n = sqrt(N0/2)*(standard_normal(s.shape)+1j*standard_normal(s.shape))
    r = s + n # received signal

    return r


# # Generate Synth Data

# In[24]:


# The overall function as illustrated by the notebooks definition at the beginning.
# NOTE: using pollution rate of 0 will return normal signals without anomalies
def generateEMSynthData(instructions, sampleRate, amountOfSignals, ampRange, flucRange, pollutionRate, pollutionLoc, 
                      snr, normalRandomSeed, anomalyRandomSeed, ifAddNoise = True):
    standard, timeTemp = generateSignal(instructions, sampleRate)
    signal = addAmplidue(standard, ampRange, instructions, sampleRate, normalRandomSeed)
    if pollutionRate > 0:
        signal = addPollution(signal, instructions, sampleRate, ampRange, pollutionRate, pollutionLoc, anomalyRandomSeed)
    
    
    signals = []
    for count in range(amountOfSignals):
        countSignal = addFluctuation(signal, flucRange)
        if ifAddNoise == True:
            countSignal = addNoise(countSignal, snr)
        signals.append(countSignal)
    
    numpySignals = np.array(signals)
    
    return numpySignals

