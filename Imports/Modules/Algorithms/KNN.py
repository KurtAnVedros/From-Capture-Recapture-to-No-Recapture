from sklearn.neighbors import NearestNeighbors
import numpy


def executeKNN(set, query, number_neighbors):

    algorithm = NearestNeighbors(n_neighbors=number_neighbors).fit(set)
    distances, indices = algorithm.kneighbors(query)
    summed_distances = numpy.sum(distances, axis=1)
    distances_list = list(summed_distances)

    return distances_list