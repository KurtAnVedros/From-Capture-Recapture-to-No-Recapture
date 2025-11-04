import numpy
import heapq
import logging


# DESCRIPTION:  Executes Cross Correlation of each signal in a set with all other signals of another set
# INPUTS:   (a) set_a - the first set of signals
#           (b) set_b - the second set of signals
#           (c) top_n - the number of most similar "neighbors"
# RETURNS:
# Note: this function is highly inefficient especially when executed in a single host environments.
# it requires NxM correlation calculations, where N is the size of set_a and M is the size of set_b
# TODO: port the execution of this function to Hadoop or Spark cluster
def executeXCorr(set_a, set_b, top_n=1):
    logging.info("executeXCorr: Executing Cross Correlation")
    logging.debug("Argument set: %s", set_a)
    logging.debug("Argument set: %s", set_b)
    logging.debug("Argument set: %s", top_n)

    strangenesses = []
    correlations = []

    # correlate each signal of a set with all other signals of another set
    # typically each signal of the training set will be "compared" with all other signals on the same set to deduce a baseline
    # then each signal on the test set will be "compared" with all other signals on the training set to estimate the deviation
    for signal_a in set_a:
        for signal_b in set_b:
            correlation = numpy.correlate(signal_a, signal_b)
            correlations.append(correlation[0])
        # choose the top N "neighboring" values i.e., the ones that have the greates similarity with the
        top_n_values = heapq.nlargest(top_n, correlations)
        correlations = []
        # the top value will always be that of the element with itself therefore we should disregard it
        strangeness = numpy.sum(top_n_values)
        strangenesses.append(strangeness)

    logging.debug("Return value strangenesses: %s", strangenesses)

    return strangenesses


def executemodXCorr(set_a, set_b, top_n=1):
    logging.info("executeXCorr: Executing Cross Correlation")
    logging.debug("Argument set: %s", set_a)
    logging.debug("Argument set: %s", set_b)
    logging.debug("Argument set: %s", top_n)

    strangenesses = []
    correlations = []

    # correlate each signal of a set with all other signals of another set
    # typically each signal of the training set will be "compared" with all other signals on the same set to deduce a baseline
    # then each signal on the test set will be "compared" with all other signals on the training set to estimate the deviation

    i = 0
    for signal_a in set_a:
        i = i + 1
        areas = []
        for signal_b in set_b:
            correlation = numpy.correlate(signal_a, signal_b, "same")
            sqcorr = numpy.square(correlation)
            areas.append(numpy.trapz(sqcorr))
        # choose the top N "neighboring" values i.e., the ones that have the greates similarity with the
        top_n_values = heapq.nlargest(top_n, areas)
        correlations = []
        # the top value will always be that of the element with itself therefore we should disregard it
        strangeness = numpy.sum(top_n_values)
        strangenesses.append(strangeness)

    logging.debug("Return value strangenesses: %s", strangenesses)

    strangenesses.sort(reverse=True)
    return strangenesses

# def sortStrangeness(strangenesses_list):

#    strangenesses_list.sort()
#    return strangenesses_list


# def calculatePValues(sorted_strangenesses_list, test_strangenesses_list):

#    p_values = []

#    m = len(test_strangenesses_list)
#    for test_element in test_strangenesses_list:
#        for b, strangeness_value in enumerate(sorted_strangenesses_list):
#            if test_element < strangeness_value:
#                break
#        p_value = (b+2)/(m+1)
#        p_values.append(p_value)
#
#
#    return p_values
