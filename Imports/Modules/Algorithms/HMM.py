from hmmlearn import hmm
from IO import Reader
import numpy

def executeHMM(base_set, query_set):

    disimilarities = []
    signal_disimilarities = []

    for base_signal in base_set:
        for i, query_signal in enumerate(query_set):

            base_signal_array = base_signal.reshape(len(base_signal), 1)
            length_base_signal_array = [len(base_signal_array)]

            query_signal_array = query_signal.reshape(len(query_signal), 1)
            length_base_signal_array = [len(query_signal_array)]

            model = hmm.GaussianHMM(n_components=3).fit(base_signal_array, length_base_signal_array)

            probability = model.score(query_signal_array)

            signal_disimilarities = signal_disimilarities + [probability]
        disimilarities.append(signal_disimilarities)
        signal_disimilarities = []

        return disimilarities