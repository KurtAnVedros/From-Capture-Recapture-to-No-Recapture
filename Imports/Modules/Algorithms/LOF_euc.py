#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from scipy.spatial import distance
import logging


# In[4]:


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


# In[7]:


# test of euc
'''
a = [1, 2, 3]
b = [1, 1, 1]

t = euc(a, b)
print(t)

a = [1, 2, 3, 4 ,5 ,6 , 7, 8, 9]
b = [1, 1, 1]

t = euc(a, b)
print(t)

a = [1, 2, 3, 4 ,5 ,6 , 7, 8, 9]
b = [1, 1, 1, 1, 1]

t = euc(a, b)
print(t)
'''


# In[5]:


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


# In[79]:


# DESCRIPTION:  Executes (a modified version of the) local outlier function to measure the strangeness between signals in two sets
# INPUTS:   (a) training_set - the set upon which the transformation will be applied without the labels
#           (b) test_set - the set of signals that is tested
#           (c) number_of_neighbors - the number of closest neighbors (more similar signals) to consider; typical value is 20
# RETURNS:  (a) base_lof - the lof for the base set
#           (b) test_lof - the lof for the test set
# NOTES:
def executeLOF(training_set, test_set, number_of_neighbors):
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
    total_proximity = euc_diss(np.concatenate((training_set, test_set)), training_set)
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

