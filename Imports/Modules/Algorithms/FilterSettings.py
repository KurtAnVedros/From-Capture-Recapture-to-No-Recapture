# imports
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
import scipy.io as sio
import csv
import os
from Algorithms import SVD
from Algorithms import SVDDenoise
from scipy.signal.signaltools import wiener
from sklearn.neighbors import KNeighborsRegressor
import warnings
from scipy.signal import savgol_filter
from scipy.spatial import distance
from sklearn.neighbors import NearestNeighbors
from skimage.restoration import denoise_wavelet

# use for weiner
# Performs a seach of the best value for noise power using the first signal the array compared to the base signal
def findBestNoisePower(signals, comparisonSignals, smallest, highest, steps):
    best = 0
    bestDist = 9999999999999999
    for n in np.arange(smallest, highest, steps):
        dist = 0
        for index in range(signals.shape[0]):
            denoised = wiener(signals[index], noise=n)
            dist = dist + distance.euclidean(denoised[index], comparisonSignals[index])
        if dist < bestDist:
            best = n
            bestDist = dist

    return best

# use for Savgol
def findBestSavgolSetting(signal1s, comparisonSignal1s, windowRange, polynomialRange):
    bestWindow = 0
    bestPolynomial = 0
    bestDist = 999999999999
    for window in windowRange:
        for polynomial in polynomialRange:
            if polynomial < window:
                with warnings.catch_warnings():
                    warnings.filterwarnings('error')
                    try:
                        dist = 0
                        denoised = np.copy(signal1s)
                        for index in range(signal1s.shape[0]):
                            denoised[index] = savgol_filter(signal1s[index], window, polynomial)
                        for index in range(denoised.shape[0]):
                            dist = dist + distance.euclidean(denoised[index], comparisonSignal1s[index])
                    except np.RankWarning:
                        dist = 9999999999999
                if dist < bestDist:
                    bestWindow = window
                    bestPolynomial = polynomial
                    bestDist = dist
    return bestWindow, bestPolynomial


def KNN_Regressor(signals, neighbor):
    clf = KNeighborsRegressor(n_neighbors=neighbor, weights='uniform')
    timeVector = np.arange(0, signals.shape[1], 1)
    timeVector = np.expand_dims(timeVector, axis=1)

    denoised = np.zeros((signals.shape[0], signals.shape[1]))
    for index in range(signals.shape[0]):
        clf.fit(timeVector, signals[index])
        denoised[index] = clf.predict(timeVector)

    return denoised


def findBestSettingsKNN_Regressor(signals, comparisonSignals, neighbors):
    bestNeighbor = -1
    bestDist = 999999999999
    for neighbor in neighbors:
        dist = 0
        denoised = KNN_Regressor(signals, neighbor)

        for index in range(denoised.shape[0]):
            dist = dist + distance.euclidean(denoised[index], comparisonSignals[index])

        if dist < bestDist:
            bestNeighbor = neighbor
            bestDist = dist

    return bestNeighbor


# SVDdenoise at a given cutting point
def SVDdenoiseCP(dataset, cuttingpoint):
    # print("Cutting Point: ", cuttingpoint)

    u, s, v_t = np.linalg.svd(dataset, full_matrices=False)

    # since the s has inf values with float64.max and float64.min values in dataset
    # s = removeInfValues1d(s)

    s_new = [value if index < cuttingpoint else 0 for index, value in enumerate(s)]
    s_new_diag = np.diag(s_new)

    denoisedDataset = np.dot(np.dot(u, s_new_diag), v_t)

    # create a 2d array of 1d arrays.
    templist = []

    for data in denoisedDataset:
        # print(np.ravel(data).shape)
        templist.append(np.ravel(data))

    templist[0].shape
    actDenoisedDataset = np.array(templist)

    return actDenoisedDataset


def findBestCuttingPoint(signals, comparisonSignals, cuttingRange):
    bestCP = -1
    bestDist = 999999999999
    for cuttingPoint in cuttingRange:
        denoised = SVDdenoiseCP(signals, cuttingPoint)
        dist = 0
        for index in range(denoised.shape[0]):
            dist = dist + distance.euclidean(denoised[index], comparisonSignals[index])
        if dist < bestDist:
            bestCP = cuttingPoint
            bestDist = dist

    return bestCP


def findBestSettingsWavelet(signals, comparisonSignals, levelRangeSearch, possibleWavelet):
    bestLevel = -1
    bestDist = 999999999999
    bestWavelet = 'temp'
    for Wavelet in possibleWavelet:
        for level in levelRangeSearch:
            dist = 0
            denoised = np.zeros((signals.shape[0], signals.shape[1]))
            for index in range(signals.shape[0]):
                denoised[index] = denoise_wavelet(signals[index], method='BayesShrink', mode='soft', wavelet_levels=level, wavelet=Wavelet,
                                       rescale_sigma='true')

            for index in range(denoised.shape[0]):
                dist = dist + distance.euclidean(denoised[index], comparisonSignals[index])

            if dist < bestDist:
                bestLevel = level
                bestDist = dist
                bestWavelet = Wavelet

    return bestLevel, bestWavelet