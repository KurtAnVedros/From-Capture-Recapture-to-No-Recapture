#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
from scipy.spatial import distance
import logging


# In[12]:


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


# In[7]:


#a = [1, 1 , 2, 2, 3, 3, 4, 4]
#b = [3, 3, 4, 4, 5, 5, 6, 6, 7, 7]

#t = dtw(a, b, 4)
#print(t)

#q = fastdtw(a, b)
#print(q)


# In[8]:


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


# In[6]:


#a = [(1, 1), (2, 2), (3, 3), (4, 4)]
#b = [(3, 3),(4, 4), (5, 5), (6, 6), (7, 7)]

#t = dtw((2,2), (7,7))
#print(t)

#test = dtw_dists(a, b, 3)
#print(test)
#print(test[0][0])

#from fastdtw import fastdtw as fDTW
#from scipy.spatial.distance import euclidean

#x = np . array ([1 , 2, 3 , 3 , 7])
#y = np . array ([1 , 2, 2 , 2 , 2 , 2 , 2, 4])
#distance , path = fDTW (x , y )

#print ( distance )
#print ( path )


# In[13]:


#a = [(1, 1), (2, 2), (3, 3), (4, 4)]
#b = [(3, 3),(4, 4), (5, 5), (6, 6)]

#print(distance.cdist(a, b, 'euclidean'))

#print(dtw_diss(a, b))


# In[79]:


def executeLOF(training_set, test_set, number_of_neighbors, window=3):
    """Compute the LOF for all base and test signals

    The simulates computing the LOF of each test signal individually,
    without corrupting the entire baseline.
    """

    logging.info("executeLOF: Executing LOF")
    logging.debug("Argument set: %s", training_set)
    logging.debug("Argument set: %s", test_set)
    logging.debug("Argument set: %s", number_of_neighbors)
    logging.debug("Argument set: %s", window)

    # build a total proximity matrix from all base and test signals to all base signals the matrix
    # is kept off balance to sort only by base signals in the future k-distances calculation
    total_proximity = dtw_dists(np.concatenate((training_set, test_set)), training_set, window)
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

