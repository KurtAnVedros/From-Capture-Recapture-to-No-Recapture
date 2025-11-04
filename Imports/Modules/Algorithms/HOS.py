import numpy
import lmoments3
import simplestatistics
import decimal


def calculateRootMeanSquare(signal):
    return simplestatistics.root_mean_square(signal.tolist())

def calculateStandardDeviation(signal):

    t = signal.tolist()
    return simplestatistics.standard_deviation(signal.tolist())


def calculateSkewness(signal):
    return simplestatistics.skew(signal.tolist())


def calculateKurtosis(signal):
    return simplestatistics.kurtosis(signal.tolist())


def calcualteLMoments(signal):
    return lmoments3._samlmusmall(signal)


def transformHOS(train_set, test_set):
    transformed_train_set = []
    for signal in train_set:
        transformed_signal = [calculateStandardDeviation(signal), calculateSkewness(signal), calculateKurtosis(signal), calculateRootMeanSquare(signal)]
        transformed_signal = transformed_signal + calcualteLMoments(signal)
        transformed_train_set.append(transformed_signal)
        transformed_signal = []

    transformed_test_set = []
    for signal in test_set:
        #transformed_signal = [calculateStandardDeviation(signal), calculateRootMeanSquare(signal)]
        transformed_signal = [calculateStandardDeviation(signal), calculateSkewness(signal), calculateKurtosis(signal), calculateRootMeanSquare(signal)]
        transformed_signal = transformed_signal + calcualteLMoments(signal)
        transformed_test_set.append(transformed_signal)
        transformed_signal = []


    return numpy.array(transformed_train_set), numpy.array(transformed_test_set)






