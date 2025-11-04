import numpy
from scipy.spatial.distance import cdist
import logging



# DESCRIPTION:  Executes (a modified version of the) local outlier function to measure the strangeness between signals in two sets
# INPUTS:   (a) training_set - the set upon which the transformation will be applied without the labels
#           (b) test_set - the set of signals that is tested
#           (c) number_of_neighbors - the number of closest neighbors (more similar signals) to consider; typical value is 20
#           (d) distance_metric - the algorithm used for estimating the distance between two signals; by default the euclidean distance
# RETURNS:  (a) base_lof - the lof for the base set
#           (b) test_lof - the lof for the test set
# NOTES:
def executeLOF(training_set, test_set, number_of_neighbors, distance_metric='euclidean'):
    """Compute the LOF for all base and test signals

    The simulates computing the LOF of each test signal individually,
    without corrupting the entire baseline.
    """

    logging.info("executeLOF: Executing LOF")
    logging.debug("Argument set: %s", training_set)
    logging.debug("Argument set: %s", test_set)
    logging.debug("Argument set: %s", number_of_neighbors)
    logging.debug("Argument set: %s", distance_metric)

    # build a total proximity matrix from all base and test signals to all base signals the matrix
    # is kept off balance to sort only by base signals in the future k-distances calculation
    total_proximity = cdist(numpy.concatenate((training_set, test_set)), training_set, distance_metric)
    # fill the proximity diaganal with infinity to prevent the sort finding identities
    numpy.fill_diagonal(total_proximity, numpy.Inf)

    # for all test and base signals sort the proximities to the k nearest base signals by index
    # the indcies allow us to reuse this matrix later for (k-distance), slice only the 'k' closest signals
    total_knn_indicies = numpy.argsort(total_proximity)[:, :number_of_neighbors]

    # use linear ndarray indexing to get all the distances by index, this creates a matrix with the
    # k nearest distances for every base and test signal to every base signal
    total_knn_distances = total_proximity.ravel()[
        total_knn_indicies + (numpy.arange(total_proximity.shape[0])[:, None] * total_proximity.shape[1])]

    # now compute the k-distance for each of the nearest neighbors for a 'k' for each signal using the saved indices
    # note that this does not return a test signal in the unlikely event that the k-distance for a base signal
    # is a particular test signal. This is to prevent polluting the distance matrix with all test signals
    total_knn_k_distances = total_knn_distances[total_knn_indicies, number_of_neighbors - 1]

    # use the maximum for the reachability distance for all of the nearest neighbors for the base and test signals
    total_reachability_distances = numpy.maximum(total_knn_distances, total_knn_k_distances)

    # compute the reachability density, simply the inverse of the average reachability distance for all neighbors
    # this flattens this to a vectoer of LRD with one value for all base and test signals
    total_local_reachability_density = number_of_neighbors / numpy.sum(total_reachability_distances, axis=1)

    # for the LOF, we can again reuse the indices to one LRD against the neighbors LRDs to get the mean
    total_lof = numpy.mean(numpy.divide(total_local_reachability_density[total_knn_indicies],
                                  total_local_reachability_density[:, None]), axis=1)

    # split the baseline and testing data up
    base_lof, test_lof = numpy.split(total_lof, [training_set.shape[0]])

    logging.debug("Return value base_lof: %s", base_lof)
    logging.debug("Return value test_lof: %s", test_lof)

    return base_lof, test_lof
