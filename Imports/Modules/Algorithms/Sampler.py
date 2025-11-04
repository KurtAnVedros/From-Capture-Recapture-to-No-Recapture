import numpy as np
from typing import List, Tuple

#Returns sampled signal produced from equally spaced samples of size no_samples from signal in a linear space
def subsample_signal(signal: np.ndarray, no_samples: int)->np.ndarray:
    """Linearly samples no_samples signals across a numpy array

    Arguments:
        signal {numpy.ndarray} -- Signal to be sampled
        no_samples {int} -- Number of samples of signal desired

    Returns:
        signal[indices] -- Sampled signal
    """
    size = len(signal)
    indices = np.floor(np.linspace(0, size, no_samples, endpoint=False)).astype(np.uint8)
    return signal[indices]


def subsample_observations(observations: np.ndarray, no_samples: int)->np.ndarray:
    """Performs subsampling on an entire dataset

    Arguments:
        observations {numpy.ndarray} -- Dataset to be subsampled
        no_samples {int} -- Number of samples

    Returns:
        result -- sampled dataset
    """
    result = []
    for observation in observations:
        result.append(subsample_signal(observation, no_samples))
    return np.array(result)

