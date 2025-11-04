import numpy as np
from typing import NewType

def find_second_max(dataset: numpy.ndarray)->int:
    """Finds the second most maximum element in the array

    Arguments:
        dataset {numpy.ndarray} -- Dataset to be searched

    Returns:
        secondmax -- int
    """
    maximum = dataset[0][0]
    secondmax = dataset[0][1]
    for i in range(len(dataset)):
        for j in range(len(dataset[i])):
            if dataset[i][j] > maximum:
                secondmax = maximum
                maximum = dataset[i][j]
            else:
                if dataset[i][j] > secondmax and dataset[i][j] != maximum:
                    secondmax = dataset[i][j]

    return secondmax


def find_second_min(dataset: numpy.ndarray)->int:
    """Finds the second most lowest number in the array

    Arguments:
        dataset {numpy.ndarray} -- Dataset to be searched

    Returns:
        secondmin -- The second-least number in the dataset
    """
    minimum = dataset[0][0]
    secondmin = dataset[0][1]
    for i in range(len(dataset)):
        for j in range(len(dataset[i])):
            if dataset[i][j] < minimum:
                secondmin = minimum
                minimum = dataset[i][j]
            else:
                if dataset[i][j] < secondmin and dataset[i][j] != minimum:
                    secondmin = dataset[i][j]

    return secondmin



def find_min(dataset: numpy.ndarray):
    """Finds the lowest number in the array

    Arguments:
        dataset {numpy.ndarray} -- Dataset to be searched

    Returns:
        minimum -- The least number in the dataset
    """
    minimum = dataset[0][0]
    for i in range(len(dataset)):
        for j in range(len(dataset[i])):
            if dataset[i][j] < minimum:
                minimum = dataset[i][j]

    return minimum