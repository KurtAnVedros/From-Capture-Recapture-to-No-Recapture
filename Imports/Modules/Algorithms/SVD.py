#!/usr/bin/env python
# coding: utf-8

# In[19]:


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
from Algorithms import FilterSettings
import random as rand


# In[20]:


def getS(dataset):
    u, s, v_t = np.linalg.svd(dataset, full_matrices=False)
    
    return s


# In[22]:


def findCutPoint(dataset, Info):
    #new_v = SVDDenoise.DeduceVhiFilter(training_set, n)
    u, s, v_t = np.linalg.svd(dataset, full_matrices=False)
    time = range(len(s))
    
    deriv = []
    for n in range(len(s)):
        if n != 0:
            deriv.append(s[n] - s[n-1])
    
    deriv2 = []
    for n in range(len(deriv)):
        if n != 0:
            if deriv[n] < 0 and deriv[n-1] < 0:
                deriv2.append(-abs(deriv[n] - deriv[n-1]))
            elif deriv[n] > 0 and deriv[n-1] > 0:
                deriv2.append(abs(deriv[n] - deriv[n-1]))
            else:
                deriv2.append((deriv[n] - deriv[n-1])) 
    
    '''
    deriv3 = []
    for n in range(len(deriv2)):
        if n != 0:
            if deriv2[n] < 0 and deriv2[n-1] < 0:
                deriv3.append(-abs(deriv2[n] - deriv2[n-1]))
            elif deriv2[n] > 0 and deriv2[n-1] > 0:
                deriv3.append(abs(deriv2[n] - deriv2[n-1]))
            else:
                deriv3.append((deriv2[n] - deriv2[n-1])) 
    '''
            
    cut = 0
    n = 0
    check = 0
    thresh = -0.5
    while check == 0:
        n = n + 1
        if deriv2[n] > thresh:
            if deriv2[n] < deriv2[n+1]:
                if deriv2[n] < deriv2[n+2]:
                    cut = n
                    check = 1
            
    cut = cut
    if Info:
        print("Cutting Point: ", cut)
    
    tanglow = cut - 10
    if tanglow < 0:
        tanglow = 0
    tanghigh = cut + 10
    temp = np.array(deriv2)
    if tanghigh > temp.shape[0]:
        tanghigh = temp.shape[0]
     
    if Info:
        print("List of tangant near Cutting Point: " + str(cut) + " (At Points " + str(tanglow) + " to " + str(tanghigh)+ ")")
    
        print(deriv2[tanglow:tanghigh])
    
    return cut


# In[23]:


def SVDdenoise(dataset, Info = False):
    cuttingpoint = findCutPoint(dataset, Info)
    
    u, s, v_t = np.linalg.svd(dataset, full_matrices=False)
    
    s_new = [value if index < cuttingpoint else 0 for index, value in enumerate(s)]
    s_new_diag = np.diag(s_new)
    
    # Shows the difference of the s and s_new
    if Info:
        plt.plot(s, label="s")
        plt.plot(s_new, '--', label="s_new")
        plt.title("S vs S_New (Cutting Point: " + str(cuttingpoint) + ") Full Graph")
        plt.xlabel("Index")
        plt.ylabel("Value")
        plt.legend()
        plt.show()

        plt.plot(s, label="s")
        plt.plot(s_new, '--', label="s_new")
        plt.title("S vs S_New (Cutting Point: " + str(cuttingpoint) + ") Zoomed Graph")
        plt.xlabel("Index")
        plt.ylabel("Value")
        
        xlimlow = cuttingpoint -10
        if xlimlow < 0:
            xlimlow = 0
        xlimhigh = cuttingpoint + 10
        plt.xlim([xlimlow, xlimhigh])
        ylimlow = s[cuttingpoint]-10
        if ylimlow < 0:
            ylimlow = 0
        ylimhigh = s[cuttingpoint]+10
        plt.ylim([ylimlow,ylimhigh])
        plt.legend()
        plt.show()
    
    denoisedDataset = np.dot(np.dot(u,s_new_diag),v_t)
    
    # create a 2d array of 1d arrays.
    templist = []
    
    for data in denoisedDataset:
        #print(np.ravel(data).shape)
        templist.append(np.ravel(data))
        
    templist[0].shape
    actDenoisedDataset = np.array(templist)
    
    return actDenoisedDataset


# In[24]:


def SVDdenoise_SAVE_S(dataset, Info = False):
    cuttingpoint = findCutPoint(dataset, Info)
    
    u, s, v_t = np.linalg.svd(dataset, full_matrices=False)
    
    s_new = [value if index < cuttingpoint else 0 for index, value in enumerate(s)]
    s_new_diag = np.diag(s_new)
    
    # Shows the difference of the s and s_new
    if Info:
        plt.plot(s, label="s")
        plt.plot(s_new, '--', label="s_new")
        plt.title("S vs S_New (Cutting Point: " + str(cuttingpoint) + ") Full Graph")
        plt.xlabel("Index")
        plt.ylabel("Value")
        plt.legend()
        plt.show()

        plt.plot(s, label="s")
        plt.plot(s_new, '--', label="s_new")
        plt.title("S vs S_New (Cutting Point: " + str(cuttingpoint) + ") Zoomed Graph")
        plt.xlabel("Index")
        plt.ylabel("Value")
        
        xlimlow = cuttingpoint -10
        if xlimlow < 0:
            xlimlow = 0
        xlimhigh = cuttingpoint + 10
        plt.xlim([xlimlow, xlimhigh])
        ylimlow = s[cuttingpoint]-10
        if ylimlow < 0:
            ylimlow = 0
        ylimhigh = s[cuttingpoint]+10
        plt.ylim([ylimlow,ylimhigh])
        plt.legend()
        plt.show()
    
    denoisedDataset = np.dot(np.dot(u,s_new_diag),v_t)
    
    # create a 2d array of 1d arrays.
    templist = []
    
    for data in denoisedDataset:
        #print(np.ravel(data).shape)
        templist.append(np.ravel(data))
        
    templist[0].shape
    actDenoisedDataset = np.array(templist)
    
    return actDenoisedDataset, s


# In[25]:


def SVDdenoise_LOAD_S(dataset, s_Saved, Info = False):
    cuttingpoint = findCutPoint(dataset, Info)
    
    u, s, v_t = np.linalg.svd(dataset, full_matrices=False)
    
    length = s.shape[0]
    
    s = s_Saved
    
    #for n in range(length - s.shape[0]):
    #    s = np.concatenate((s, [0]))
    
    s_new = [value if index < cuttingpoint else 0 for index, value in enumerate(s)]
    s_new_diag = np.diag(s_new)
    
    # Shows the difference of the s and s_new
    if Info:
        plt.plot(s, label="s")
        plt.plot(s_new, '--', label="s_new")
        plt.title("S vs S_New (Cutting Point: " + str(cuttingpoint) + ") Full Graph")
        plt.xlabel("Index")
        plt.ylabel("Value")
        plt.legend()
        plt.show()

        plt.plot(s, label="s")
        plt.plot(s_new, '--', label="s_new")
        plt.title("S vs S_New (Cutting Point: " + str(cuttingpoint) + ") Zoomed Graph")
        plt.xlabel("Index")
        plt.ylabel("Value")
        
        xlimlow = cuttingpoint -10
        if xlimlow < 0:
            xlimlow = 0
        xlimhigh = cuttingpoint + 10
        plt.xlim([xlimlow, xlimhigh])
        ylimlow = s[cuttingpoint]-10
        if ylimlow < 0:
            ylimlow = 0
        ylimhigh = s[cuttingpoint]+10
        plt.ylim([ylimlow,ylimhigh])
        plt.legend()
        plt.show()
    
    denoisedDataset = np.dot(np.dot(u,s_new_diag),v_t)
    
    # create a 2d array of 1d arrays.
    templist = []
    
    for data in denoisedDataset:
        #print(np.ravel(data).shape)
        templist.append(np.ravel(data))
        
    templist[0].shape
    actDenoisedDataset = np.array(templist)
    
    return actDenoisedDataset


# In[ ]:


def SVDdenoiseCP_LOAD_S(dataset, s_Saved, cuttingpoint, Info = False):
    
    u, s, v_t = np.linalg.svd(dataset, full_matrices=False)
    
    length = s.shape[0]
    
    s = s_Saved
    
    #for n in range(length - s.shape[0]):
    #    s = np.concatenate((s, [0]))
    
    s_new = [value if index < cuttingpoint else 0 for index, value in enumerate(s)]
    s_new_diag = np.diag(s_new)
    
    # Shows the difference of the s and s_new
    if Info:
        plt.plot(s, label="s")
        plt.plot(s_new, '--', label="s_new")
        plt.title("S vs S_New (Cutting Point: " + str(cuttingpoint) + ") Full Graph")
        plt.xlabel("Index")
        plt.ylabel("Value")
        plt.legend()
        plt.show()

        plt.plot(s, label="s")
        plt.plot(s_new, '--', label="s_new")
        plt.title("S vs S_New (Cutting Point: " + str(cuttingpoint) + ") Zoomed Graph")
        plt.xlabel("Index")
        plt.ylabel("Value")
        
        xlimlow = cuttingpoint -10
        if xlimlow < 0:
            xlimlow = 0
        xlimhigh = cuttingpoint + 10
        plt.xlim([xlimlow, xlimhigh])
        ylimlow = s[cuttingpoint]-10
        if ylimlow < 0:
            ylimlow = 0
        ylimhigh = s[cuttingpoint]+10
        plt.ylim([ylimlow,ylimhigh])
        plt.legend()
        plt.show()
    
    denoisedDataset = np.dot(np.dot(u,s_new_diag),v_t)
    
    # create a 2d array of 1d arrays.
    templist = []
    
    for data in denoisedDataset:
        #print(np.ravel(data).shape)
        templist.append(np.ravel(data))
        
    templist[0].shape
    actDenoisedDataset = np.array(templist)
    
    return actDenoisedDataset


# In[26]:


# SVDdenoise at a given cutting point
def SVDdenoiseCP(dataset, cuttingpoint, Info):    
    print("Cutting Point: ", cuttingpoint)
    
    u, s, v_t = np.linalg.svd(dataset, full_matrices=False)
    
    s_new = [value if index < cuttingpoint else 0 for index, value in enumerate(s)]
    s_new_diag = np.diag(s_new)
    
    if Info:
        plt.plot(s, label="s")
        plt.plot(s_new, '--', label="s_new")
        plt.title("S vs S_New (Cutting Point: " + str(cuttingpoint) + ") Full Graph")
        plt.xlabel("Index")
        plt.ylabel("Value")
        plt.legend()
        plt.show()

        plt.plot(s, label="s")
        plt.plot(s_new, '--', label="s_new")
        plt.title("S vs S_New (Cutting Point: " + str(cuttingpoint) + ") Zoomed Graph")
        plt.xlabel("Index")
        plt.ylabel("Value")

        xlimlow = cuttingpoint -10
        if xlimlow < 0:
            xlimlow = 0
        xlimhigh = cuttingpoint + 10
        plt.xlim([xlimlow, xlimhigh])
        ylimlow = s[cuttingpoint]-10
        if ylimlow < 0:
            ylimlow = 0
        ylimhigh = s[cuttingpoint]+10
        plt.ylim([ylimlow,ylimhigh])
        plt.legend()
        plt.show()
    
    denoisedDataset = np.dot(np.dot(u,s_new_diag),v_t)
    
    # create a 2d array of 1d arrays.
    templist = []
    
    for data in denoisedDataset:
        #print(np.ravel(data).shape)
        templist.append(np.ravel(data))
        
    templist[0].shape
    actDenoisedDataset = np.array(templist)
    
    return actDenoisedDataset


# In[27]:


# SVDdenoise at a given cutting point
def SVDdenoiseCPwithNandA(normal, anomaly, cuttingpoint, Info):
    length = normal.shape[0]
    # concatenate
    snrTotal = np.concatenate((normal, anomaly))
    # perform SVD
    snrTotalSVD = SVDdenoiseCP(snrTotal, cuttingpoint, Info)
    # separate
    normal_SVD = snrTotalSVD[:length]
    anomaly_SVD = snrTotalSVD[length: 2*length]
    
    return normal_SVD, anomaly_SVD


# In[28]:


# SVDdenoise at a given cutting point
def SVDdenoisewithNandA(normal, anomaly, Info):
    length = normal.shape[0]
    # concatenate
    snrTotal = np.concatenate((normal, anomaly))
    # perform SVD
    snrTotalSVD = SVDdenoise(snrTotal)
    # separate
    normal_SVD = snrTotalSVD[:length]
    anomaly_SVD = snrTotalSVD[length: 2*length]
    
    return normal_SVD, anomaly_SVD


# In[29]:


# SVDdenoise at a given cutting point
def SVDdenoisewithNandA_SAVED_S(normal, anomaly, cuttingpoint, Info):
    length = normal.shape[0]
    # concatenate
    snrTotal = np.concatenate((normal, anomaly))
    # perform SVD
    snrTotalSVD, s = SVDdenoise_SAVE_S(snrTotal, cuttingpoint, Info)
    # separate
    normal_SVD = snrTotalSVD[:length]
    anomaly_SVD = snrTotalSVD[length: 2*length]
    
    return normal_SVD, anomaly_SVD, s


# In[30]:


# SVDdenoise at a given cutting point
def SVDdenoisewithNandA_LOAD_S(normal, anomaly, cuttingpoint, s,  Info):
    length = normal.shape[0]
    # concatenate
    snrTotal = np.concatenate((normal, anomaly))
    # perform SVD
    snrTotalSVD = SVDdenoise_LOAD_S(snrTotal, cuttingpoint, Info)
    # separate
    normal_SVD = snrTotalSVD[:length]
    anomaly_SVD = snrTotalSVD[length: 2*length]
    
    return normal_SVD, anomaly_SVD

