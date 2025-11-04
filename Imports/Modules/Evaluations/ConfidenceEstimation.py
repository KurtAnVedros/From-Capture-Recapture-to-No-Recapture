import pylab
from sklearn import metrics
import math
from sklearn.metrics import auc
import numpy
import logging



# DESCRIPTION:  Calculates some basic metrics that can be deduced by a confusion matrix
#
# INPUTS:   (a) ground_truth - the real class/labels (usually represented as the y vector)
#           (b) predictions - the predicted values produced from a classification process
# RETURNS:  total_metrics - a dictionary with the basic metrics
# NOTES: For some metrics, when the denominator to be generated is 0, NaN will be returned instead.
#        This may create unwanted behavior in the future.
def calculateBasicMetrics(ground_truth, predictions):

    logging.info("calculateBasicMetrics: Calculating basic metrics")
    logging.debug("Argument ground_truth: %s", ground_truth)
    logging.debug("Argument predictions: %s", predictions)

    test = metrics.confusion_matrix(ground_truth, predictions)
    tn, fp, fn, tp = metrics.confusion_matrix(ground_truth, predictions, labels=[0,1]).ravel()

    total_metrics = {
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "sensitivity": tp/(tp+fn) if tp+fn > 0 else float('NaN'),
        "specificity": tn/(tn+fp) if tn+fp > 0 else float('NaN'),
        "precision": tp/(tp+fp) if tp+fp > 0 else float('NaN'),
        "negative_predictive_value": tn/(tn+fn) if (tn+fn) > 0 else float('NaN'),
        "fall_out": 1 - (tn/(tn+fp)) if (tn+fp) > 0 else float('NaN'),
        "false_negative_rate": 1 - (tp/(tp+fn)) if (tp+fn) > 0 else float('NaN'),
        "false_discovery_rate": 1 - (tp/(tp+fp)) if (tp+fp) > 0 else float('NaN'),
        "acc": (tp+tn)/(tp+fp+fn+tn) if (tp+fp+fn+tn) > 0 else float('NaN'),
        "f1": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else float('NaN'),
        "matthews": (tp * tn - fp * fn) / math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)) if (math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))) > 0 else float('NaN'),
        "informedness": (tp/(tp+fn)) + (tn/(tn+fp)) - 1 if (tp+fn) > 0 and (tn+fp) > 0 else float('NaN'),
        "markedness": (tp/(tp+fp)) + (tn/(tn+fn)) - 1 if (tp+fp) > 0 and (tn+fn) > 0 else float('NaN')

    }

    logging.debug("Return value total_metrics: %s", total_metrics)

    return total_metrics


# DESCRIPTION:  Calculates the accumulative results based on a series of a metrics
#
# INPUTS:   (a) list_of_metrics - a list containing several metrics dictionaries
# RETURNS:  (a) results - a dictionary with the accumulative results e.g., mean accuracy AUC etc.
# NOTES: This function may need to be extended with additional metrics if there is need for it, in the future
def calculateMultipleRoundMetrics(list_of_metrics):

    logging.info("calculateMultipleRoundMetrics: Calculating basic metrics")
    logging.debug("Argument list_of_metrics: %s", list_of_metrics)

    fprs_list = [item["fall_out"] for item in list_of_metrics]
    tprs_list = [item["sensitivity"] for item in list_of_metrics]
    auc_value = auc(fprs_list, tprs_list)
    mean_acc = numpy.mean([item["acc"] for item in list_of_metrics])

    total_metrics = {
        "fprs": fprs_list,
        "tprs": tprs_list,
        "auc": auc_value,
        "mean_acc": mean_acc
    }

    logging.debug("Return value total_metrics: %s", total_metrics)

    return total_metrics


# DESCRIPTION:  Evaluates p-values against ground truth in a range of confidence values
#
# INPUTS:   (a) test_p_values - a list containing the p-values
#           (b) ground_truth - a list containing
#           (c) range_start - start of the range of a the confidence value to be taken into account
#           (d) range_end - end of the range of the confidence value to be taken into account
#           (e) step - the increment step for the confidence
# RETURNS:  (a) results - a list of dictionaries containing the basic metrics
# NOTES: This function may need to be extended with additional metrics if there is need for it, in the future
def getEvaluation(test_p_values, ground_truth, range_start, range_end, step):

    logging.info("getEvaluation: Evaluating p-values against ground truth")
    logging.debug("Argument test_p_values: %s", test_p_values)
    logging.debug("Argument ground_truth: %s", ground_truth)
    logging.debug("Argument range_start: %s", range_start)
    logging.debug("Argument range_end: %s", range_end)
    logging.debug("Argument step: %s", step)

    results = []

    #for conf in pylab.frange(range_start, range_end, step):
    for conf in numpy.arange(range_start, range_end, step):
        predictions = []
        for p in test_p_values:
            if p <= 1 - conf:
                predictions.append(1)
            else:
                predictions.append(0)

        metrics = calculateBasicMetrics(ground_truth, predictions)
        results.append(metrics)


    #logging.debug("Return value results: %s", results)

    return results


# DESCRIPTION:  Evaluates multiple p-values against ground truth in a range of confidence values,
# This method is used when multiple training sets are given and calculated p-values of each are taken. 
#
# INPUTS:   (a) test_p_values - a 2D numppy array containing the p-values
#           (b) ground_truth - a list containing
#           (c) range_start - start of the range of a the confidence value to be taken into account
#           (d) range_end - end of the range of the confidence value to be taken into account
#           (e) step - the increment step for the confidence
# RETURNS:  (a) results - a list of dictionaries containing the basic metrics
# NOTES: This function may need to be extended with additional metrics if there is need for it, in the future
def getEvaluationMultiTrain(test_p_values, ground_truth, range_start, range_end, step):

    logging.info("getEvaluation: Evaluating p-values against ground truth")
    logging.debug("Argument test_p_values: %s", test_p_values)
    logging.debug("Argument ground_truth: %s", ground_truth)
    logging.debug("Argument range_start: %s", range_start)
    logging.debug("Argument range_end: %s", range_end)
    logging.debug("Argument step: %s", step)

    results = []

    #for conf in pylab.frange(range_start, range_end, step):
    for conf in numpy.arange(range_start, range_end, step):
        

        predictions = []
        # go through p-values points
        for p in range(test_p_values.shape[1]):
            # Go through each training set p-values giving a vote.
            vote = 0
            
            for a in range(test_p_values.shape[0]):
                if test_p_values[a][p] <= 1 - conf:
                    vote = vote + 1
            
            if vote == test_p_values.shape[0]:
                predictions.append(1)              
            else:
                predictions.append(0)         
                
                
        metrics = calculateBasicMetrics(ground_truth, predictions)
        results.append(metrics)


    #logging.debug("Return value results: %s", results)

    return results


def getEnsembleEvaluation(test_p1_values, test_p2_values, test_p3_values, ground_truth, range_start, range_end, step):
    logging.info("getEvaluation: Evaluating p-values against ground truth")
    logging.debug("Argument test_p_values: %s", test_p1_values)
    logging.debug("Argument ground_truth: %s", ground_truth)
    logging.debug("Argument range_start: %s", range_start)
    logging.debug("Argument range_end: %s", range_end)
    logging.debug("Argument step: %s", step)

    results = []

    for conf in pylab.frange(range_start, range_end, step):
        predictions = []
        for i, p in enumerate(test_p1_values):
            votes = 0
            if test_p1_values[i] <= 1 - conf:
                votes = votes + 1
            if test_p2_values[i] <= 1 - conf:
                votes = votes + 1
            if test_p3_values[i] <= 1 - conf:
                votes = votes + 1

            if votes >= 2:
                predictions.append(1)
            else:
                predictions.append(0)



        metrics = calculateBasicMetrics(ground_truth, predictions)
        results.append(metrics)

        logging.debug("Return value results: %s", results)

    return results
