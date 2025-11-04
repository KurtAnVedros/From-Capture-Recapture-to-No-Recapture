#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
from scipy.spatial import distance
import logging
import math

from frechetdist import frdist


# # DTW

# In[3]:


# DESCRIPTION:  Executes the Dynamic Time Warping DTW function of two time series used as the dissimilar metric. This function can be used as the distance metric.
# INPUTS:   (a) x - time series to compare too
#           (b) y - time series to compare with
#           (c) window - The wraping window to limit the amount of times the path can travel in one dirrection.
# RETURNS:  (a) The DTW dissimilarity
# NOTE: Modified code from https://towardsdatascience.com/dynamic-time-warping-3933f25fcdd
# NOTE: issue in the fact that a large array needs to be allocated to utalize.
def dtw(x, y, window = 3, matrix = False):
    n, m = len(x), len(y)
    w = np.max([window, abs(n-m)])
    dtw_matrix = np.zeros((n+1, m+1), dtype=np.float16)
    
    for i in range(n+1):
        for j in range(m+1):
            dtw_matrix[i, j] = np.inf
    dtw_matrix[0, 0] = 0
    
    for i in range(1, n+1):
        for j in range(np.max([1, i-w]), np.min([m, i+w])+1):
            dtw_matrix[i, j] = 0
    
    for i in range(1, n+1):
        for j in range(np.max([1, i-w]), np.min([m, i+w])+1):
            cost = abs(x[i-1] - y[j-1])
            # take the min from the square box.
            last_min = np.min([dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1]])
            dtw_matrix[i, j] = cost + last_min
    
    dtw_dissimilarity = dtw_matrix[n, m]
    # NOTE: can return the matrix as well.
    if matrix == True:
        return dtw_dissimilarity, dtw_matrix
    else:
        return dtw_dissimilarity
    
## Alternative faster DTW that gives approximates
from fastdtw import fastdtw as fDTW
from scipy.spatial.distance import euclidean

def fastdtw(x, y, matrix = False):
    dtw_dissimilarity, dtw_path = fDTW(x , y)

    return dtw_dissimilarity


# In[4]:


# DESCRIPTION:  Executes (a modified version of the) local outlier function to measure the strangeness between signals in two sets
# INPUTS:   (a) XA - the list of signals to compare to
#           (b) XB - the list of signals to compare with
#           (c) window - The wraping window to limit the amount of times the path can travel in one dirrection.
# RETURNS:  (a) the dtw dissimilarity measure
def dtw_dists(XA, XB, window = 3):
    x = len(XA)
    y = len(XB)

    dissimilarity = np.zeros((x,y))
    xaCount = -1
    for rowXA in XA:
        xbCount = -1
        xaCount = xaCount+1
        for rowXB in XB:
            xbCount = xbCount+1
            dissimilarity[xaCount][xbCount] = fastdtw(rowXA, rowXB)
    
    return dissimilarity


# # Euc

# In[15]:


# DESCRIPTION:  Executes euclidean function of two time series used as the distance metric. .
# INPUTS:   (a) x - time series to compare too
#           (b) y - time series to compare with
# RETURNS:  (a) The euclidean distance of the signals limiting them by the shortest signal length
# NOTE: example if x length is 450 and y length is 400, will perform euclidean distance of x[0:400] y[0:400]
def euc(x, y):
    shortest = len(x)
    if len(y) < shortest:
        shortest = len(y)
    xLimit = np.array(x[:shortest])
    yLimit = np.array(y[:shortest])
    return np.sqrt(np.sum(np.square(xLimit - yLimit)))


# In[6]:


# DESCRIPTION:  Executes Euclidean distance measure on the entire two arrays.
# INPUTS:   (a) XA - the list of signals to compare to
#           (b) XB - the list of signals to compare with
# RETURNS:  (a) the euclidean distance measure
def euc_diss(XA, XB):
    x = len(XA)
    y = len(XB)

    dissimilarity = np.zeros((x,y))
    xaCount = -1
    for rowXA in XA:
        xbCount = -1
        xaCount = xaCount+1
        for rowXB in XB:
            xbCount = xbCount+1           
            dissimilarity[xaCount][xbCount] = euc(rowXA, rowXB)
    
    return dissimilarity


# # CID

# In[12]:


# DESCRIPTION:  Executes Complexity Invariant Distance on the two arrays. Note: use this with other distance measures by
#                multiplication.
# INPUTS:   (a) XA - the list of signals to compare to
#           (b) XB - the list of signals to compare with
# RETURNS:  (a) the complexity invariant distance
def CID(x, y):
    CEDx = math.sqrt(np.sum(pow(np.diff(x),2)))
    CEDy = math.sqrt(np.sum(pow(np.diff(y),2)))
    CF = max(CEDx, CEDy)/min(CEDx, CEDy)
    return CF


# In[8]:


def cideuc_diss(XA, XB):
    x = len(XA)
    y = len(XB)

    dissimilarity = np.zeros((x,y))
    xaCount = -1
    for rowXA in XA:
        xbCount = -1
        xaCount = xaCount+1
        for rowXB in XB:
            xbCount = xbCount+1  
            dissimilarity[xaCount][xbCount] = euc(rowXA, rowXB) * CID(rowXA, rowXB)

    return dissimilarity


# In[9]:


def ciddtw_diss(XA, XB, window = 3):
    x = len(XA)
    y = len(XB)

    dissimilarity = np.zeros((x,y))
    xaCount = -1
    for rowXA in XA:
        xbCount = -1
        xaCount = xaCount+1
        for rowXB in XB:
            xbCount = xbCount+1
            dissimilarity[xaCount][xbCount] = fastdtw(rowXA, rowXB) * CID(rowXA, rowXB)
    
    return dissimilarity


# # Frechet

# In[31]:


def convert1dto2d(arr):
    arr2d = []
    count = 0
    for i in arr:
        arr2d.append([count, i])
        count = count + 1
    return arr2d
    

def frech(x, y):
    if x.ndim == 1:
        x = convert1dto2d(x)
    if y.ndim == 1:
        y = convert1dto2d(y)
    print(x)
    print(y)
    return frdist(x,y)

def frech_diss(XA, XB):
    x = len(XA)
    y = len(XB)

    dissimilarity = np.zeros((x,y))
    xaCount = -1
    for rowXA in XA:
        xbCount = -1
        xaCount = xaCount+1
        for rowXB in XB:
            xbCount = xbCount+1  
            dissimilarity[xaCount][xbCount] = frech(rowXA, rowXB)
        
    return dissimilarity


# # LOF 

# In[5]:


# DESCRIPTION:  Executes (a modified version of the) local outlier function to measure the strangeness between signals in two sets
# INPUTS:   (a) training_set - the set upon which the transformation will be applied without the labels
#           (b) test_set - the set of signals that is tested
#           (c) number_of_neighbors - the number of closest neighbors (more similar signals) to consider; typical value is 20
#           (d) dissimialrity - dissimilarity or distance measure.
#                  euc, dtw, cidecu, ciddtw
#           (e) Window for dtw dissimialrity algorithm
# RETURNS:  (a) base_lof - the lof for the base set
#           (b) test_lof - the lof for the test set
# NOTES:
def executeLOF(training_set, test_set, number_of_neighbors, dissimilarity, window = 3):
    """Compute the LOF for all base and test signals

    The simulates computing the LOF of each test signal individually,
    without corrupting the entire baseline.
    """

    logging.info("executeLOF: Executing LOF")
    logging.debug("Argument set: %s", training_set)
    logging.debug("Argument set: %s", test_set)
    logging.debug("Argument set: %s", number_of_neighbors)

    # build a total proximity matrix from all base and test signals to all base signals the matrix
    # is kept off balance to sort only by base signals in the future k-distances calculation
    if dissimilarity == "euc":
        total_proximity = euc_diss(np.concatenate((training_set, test_set)), training_set)
    if dissimilarity == "dtw":
        total_proximity = dtw_dists(np.concatenate((training_set, test_set)), training_set, window)
    if dissimilarity == "cideuc":
        total_proximity = cideuc_diss(np.concatenate((training_set, test_set)), training_set)
    if dissimilarity == "ciddtw":
        total_proximity = ciddtw_diss(np.concatenate((training_set, test_set)), training_set)
    if dissimilarity == "frech":
        total_proximity = frech_diss(np.concatenate((training_set, test_set)), training_set)
    # fill the proximity diaganal with infinity to prevent the sort finding identities
    np.fill_diagonal(total_proximity, np.Inf)

    # for all test and base signals sort the proximities to the k nearest base signals by index
    # the indcies allow us to reuse this matrix later for (k-distance), slice only the 'k' closest signals
    total_knn_indicies = np.argsort(total_proximity)[:, :number_of_neighbors]

    # use linear ndarray indexing to get all the distances by index, this creates a matrix with the
    # k nearest distances for every base and test signal to every base signal
    total_knn_distances = total_proximity.ravel()[
        total_knn_indicies + (np.arange(total_proximity.shape[0])[:, None] * total_proximity.shape[1])]

    # now compute the k-distance for each of the nearest neighbors for a 'k' for each signal using the saved indices
    # note that this does not return a test signal in the unlikely event that the k-distance for a base signal
    # is a particular test signal. This is to prevent polluting the distance matrix with all test signals
    total_knn_k_distances = total_knn_distances[total_knn_indicies, number_of_neighbors - 1]

    # use the maximum for the reachability distance for all of the nearest neighbors for the base and test signals
    total_reachability_distances = np.maximum(total_knn_distances, total_knn_k_distances)

    # compute the reachability density, simply the inverse of the average reachability distance for all neighbors
    # this flattens this to a vectoer of LRD with one value for all base and test signals
    total_local_reachability_density = number_of_neighbors / np.sum(total_reachability_distances, axis=1)

    # for the LOF, we can again reuse the indices to one LRD against the neighbors LRDs to get the mean
    total_lof = np.mean(np.divide(total_local_reachability_density[total_knn_indicies],
                                  total_local_reachability_density[:, None]), axis=1)

    # split the baseline and testing data up
    base_lof, test_lof = np.split(total_lof, [training_set.shape[0]])

    logging.debug("Return value base_lof: %s", base_lof)
    logging.debug("Return value test_lof: %s", test_lof)

    return base_lof, test_lof


# In[ ]:




