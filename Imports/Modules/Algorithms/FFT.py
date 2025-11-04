from scipy.fftpack import rfft
from scipy.fftpack import fftfreq
from typing import Tuple
import logging
import numpy
import numpy as np

# DESCRIPTION:  Executes Fast Fourier Transformation on given training and testing datasets
# INPUTS:   (a) train_set - the training set upon which the transformation will be applied without the labels
#           (b) test_set - the test set upon which the the transformation will be applied without the labels
# RETURNS:  (a) transformed_training_set - a numpy array with the values of the transformed training set
#           (b) test_set - a numpy array with the values of the transformed test set
# NOTES: test_size should be a value between 0 and 1;
def executeFFT(train_set, test_set):

    logging.info("executeFFT: Executing FFT")
    logging.debug("Argument train_set: %s", train_set)
    logging.debug("Argument test_set: %s", test_set)

    transformed_training_set = rfft(train_set)
    transformed_test_set = rfft(test_set)


    logging.debug("Return value transformed_training_set: %s", transformed_training_set)
    logging.debug("Return value transformed_test_set: %s", transformed_test_set)

    return transformed_training_set, transformed_test_set

#def executeFFTAmplitude(train_set: numpy.ndarray, test_set: numpy.ndarray)>Tuple[numpy.ndarray]:
def executeFFTAmplitude(train_set: numpy.ndarray, test_set: numpy.ndarray):
    """Does a FFT amplitude one-sidd transform on two input datasets

    Arguments:
        train_set {numpy.ndarray} -- The first set to be FFT amplitude transformed
        test_set {numpy.ndarray} -- The second dataset to be FFT amplitude transformed

    Returns:
        (train_amplitudes, test_amplitudes) -- Tuple containing transformed amplitudes of two datasets
    """
    n=len(train_set[0])
    train_amplitudes=(2/n)*np.abs(np.fft.fft(train_set))
    test_amplitudes=(2/n)*np.abs(np.fft.fft(test_set))
    return (train_amplitudes,test_amplitudes)





def getFrequencies(fft_signal: numpy.ndarray, sample_interval: int):

    n = len(fft_signal)

    fft_signal_f = fftfreq(n, sample_interval)
    mask = numpy.where(fft_signal_f >= 0)

    xx = fft_signal_f[mask]
    yy = abs(fft_signal[mask]) / n


    return xx, yy


def getFrequencies2(fft_signal, sample_rate):

     N = len(fft_signal)
     T = 1.0 / sample_rate
     yf = 2.0 / N * numpy.abs(fft_signal[0:N // 2])
     xf = numpy.linspace(0.0, 1.0 / (2.0 * T), N // 2)

     return xf, yf





