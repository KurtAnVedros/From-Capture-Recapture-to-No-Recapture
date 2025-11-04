import numpy as np
from secrets import randbelow


def shuffle(signal: numpy.ndarray):
    """Switches two parts of an array at a random index and rotates them

    Arguments:
        signal {numpy.ndarray} -- signal to be shuffled

    Returns:
        shuffled signal
    """
    sz = len(signal)
    end_index = randbelow(sz)
    chunk = signal[0:end_index]
    rest = signal[end_index:sz]
    return np.concatenate((rest, chunk), axis=0)


def shuffle_signals(signals: numpy.ndarray):
    """Shuffles an array of signals

    Arguments:
        signals {numpy.ndarray} -- Signals to be shuffled

    Returns:
        shuffled_signals
    """
    shuffled_signals = np.zeros_like(signals)
    for i in range(len(signals)):
        shuffled_signals[i] = shuffle(signals[i])
    return shuffled_signals
