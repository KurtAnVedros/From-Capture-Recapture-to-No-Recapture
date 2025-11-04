import matplotlib.pyplot
#matplotlib.use('Agg')
from IO import Writer
import logging
import numpy
import sklearn.metrics as metrics


# DESCRIPTION:  Creates and saves a plot illustrating a ROC curve
#
# INPUTS:   (a) fprs - false positive rate for the Y axis
#           (b) tprs - true positive rate for the X axis
#           (c) color - color of the line
#           (d) lw - the weight of the line
#           (e) label - the label of legend that corresponds to the line
#           (f) x_label - the label of the X axis
#           (g) y_label - the label of the Y axis
#           (h) title - the title of the figure
#           (i) legend_location - the location of the legend that corresponds to the line
#           (j) auc - the auc score to be added to legend
#           (k) output_path - the file path for the resulting image
# RETURNS:  (a) results - a list of dictionaries containing the basic metrics
# NOTES: -
def plotROC(fprs, tprs, color, lw, label, x_label, y_label, title, legend_location, auc=None, output_path=None):

    logging.info("plotROC: Generating ROC graph")
    logging.debug("Argument fprs: %s", fprs)
    logging.debug("Argument tprs: %s", tprs)
    logging.debug("Argument color: %s", color)
    logging.debug("Argument lw: %s", lw)
    logging.debug("Argument label: %s", label)
    logging.debug("Argument x_label: %s", x_label)
    logging.debug("Argument y_label: %s", y_label)
    logging.debug("Argument title: %s", title)
    logging.debug("Argument legend_location: %s", legend_location)
    logging.debug("Argument auc: %s", auc)
    logging.debug("Argument output_path: %s", output_path)

    matplotlib.pyplot.figure()
    #if the auc value is provided concatenate it to the label (legend) of the line
    if auc is not None:
        label = label + " (AUC: {:0.3f})".format(auc)
    matplotlib.pyplot.plot(fprs, tprs, color=color, lw=lw, label=label)
    #plot the worst case scenario as a gray dashed line
    matplotlib.pyplot.plot([0, 1], [0, 1], color="gray", lw=2, linestyle="--")
    matplotlib.pyplot.xlim([0.0, 1.0])
    #increase the limits of the Y axis slightly beyond 1 to look better
    matplotlib.pyplot.ylim([0.0, 1.05])
    matplotlib.pyplot.xlabel(x_label)
    matplotlib.pyplot.ylabel(y_label)
    matplotlib.pyplot.title(title)
    matplotlib.pyplot.legend(loc=legend_location)
    if output_path is not None:
        try:
            Writer.savePlot(output_path)
        except FileNotFoundError as error:
            logging.error(error)
    else:
        matplotlib.pyplot.show()




# DESCRIPTION: Creates and saves a plot illustrating multiple ROC curve data
#
# INPUTS:   (a) fprsies - numpy array of false positive rate for the Y axis for each fold
#           (b) tprsies - numpy array of true positive rate for the X axis for each fold
#           (c) aucs - numpy array of all the AUCs for each fold
#           (d) lw - the weight of the line
#           (f) x_label - the label of the X axis
#           (g) y_label - the label of the Y axis
#           (h) title - the title of the figure
#           (i) legend_location - the location of the legend that corresponds to the line
#           (k) output_path - the file path for the resulting image
# RETURNS:  (a) results - a list of dictionaries containing the basic metrics
def plotROCMultiple(fprsies, tprsies, aucs, tprs_mean, color, lw, x_label, y_label, title, legend_location, output_path=None):
    logging.info("plotROC: Generating ROC graph")
    logging.debug("Argument fprsies: %s", fprsies)
    logging.debug("Argument tprsies: %s", tprsies)
    logging.debug("Argument aucs: %s", aucs)
    logging.debug("Argument aucs: %s", tprs_mean)
    logging.debug("Argument color: %s", color)
    logging.debug("Argument lw: %s", lw)
    logging.debug("Argument x_label: %s", x_label)
    logging.debug("Argument y_label: %s", y_label)
    logging.debug("Argument title: %s", title)
    logging.debug("Argument legend_location: %s", legend_location)
    logging.debug("Argument output_path: %s", output_path)
    
    # get the highest AUC
    low_index = 0
    high_index = 0
    
    lowest = 10000000
    highest = 0
    
    fprsies = numpy.array(fprsies)
    tprsies = numpy.array(tprsies)
    aucs = numpy.array(aucs)
    tprs_mean = numpy.array(tprs_mean)
    
    
    for index in range(aucs.shape[0]):
        if aucs[index] >= highest:
            highest = aucs[index]
            high_index = index
        if aucs[index] <= lowest:
            lowest = aucs[index]
            low_index = index
    
    # get the highest AUC
    fprs_high = fprsies[high_index]
    tprs_high = tprsies[high_index]
    auc_high = aucs[high_index]
    
    # get the lowest AUC
    fprs_low = fprsies[low_index]
    tprs_low = tprsies[low_index]
    auc_low = aucs[low_index]
    
    #get average AUC
    
    #interp_tprs = []
    #for fold in range(aucs.shape[0]):
    #    fpr = fprsies[fold]
    #    tpr = tprsies[fold]
    #    interp_tpr = numpy.interp(mean_fpr, fpr,tpr)
    #    interp_tpr[0] = 0.0
    #    interp_tprs.append(interp_tpr)
    
    #interp_tprs = numpy.array(interp_tprs)
     
    #print(interp_tprs.shape)
    #print(interp_tprs)
    
    #mean_fpr = numpy.flip(mean_fpr,0)
    #mean_fpr = numpy.mean(fprsies, axis = 0)
    mean_tpr = tprs_mean.mean(axis = 0)
    base_fpr = numpy.linspace(0, 1, mean_tpr.shape[0])
    #mean_tpr[-1] = 1.0
    #print(mean_tpr.shape)
    #print(mean_tpr)
    print(base_fpr.shape)
    print(mean_tpr.shape)
    std_tpr = tprs_mean.std(axis=0)
    if auc_low == 1 and auc_high == 1:
        mean_auc == auc_high
    else:    
        mean_auc = metrics.auc(base_fpr, mean_tpr)
    std_auc = numpy.std(tprs_mean)
    
    
    
    matplotlib.pyplot.fill_between(base_fpr, tprs_mean[low_index] , tprs_mean[high_index], color='grey', alpha=.2)
    
    matplotlib.pyplot.plot(fprs_high, tprs_high, color='blue', lw=lw, 
                               label=r'High ROC (AUC = %0.3f)' % (auc_high))
    
    matplotlib.pyplot.plot(base_fpr, mean_tpr, color=color, lw=lw, linestyle ="--", 
                               label=r'Mean ROC (AUC = %0.3f)' % (mean_auc))
    
    matplotlib.pyplot.plot(fprs_low, tprs_low, color='red', lw=lw, 
                               label=r'Low ROC (AUC = %0.3f)' % (auc_low))
    
    std_tpr = numpy.std(tprsies, axis=0)
    tprs_upper = numpy.minimum(mean_tpr + std_tpr, 1)
    #tprs_upper = numpy.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = numpy.maximum(mean_tpr - std_tpr, 0)
    #tprs_lower = mean_tpr-std_tpr
    #matplotlib.pyplot.fill_between(base_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')
    
    matplotlib.pyplot.plot([0, 1], [0, 1], color="gray", lw=2, linestyle="--")
    matplotlib.pyplot.xlim([0.0, 1.0])
    #increase the limits of the Y axis slightly beyond 1 to look better
    matplotlib.pyplot.ylim([0.0, 1.05])
    matplotlib.pyplot.xlabel(x_label)
    matplotlib.pyplot.ylabel(y_label)
    #matplotlib.pyplot.title(title)
    matplotlib.pyplot.legend(loc=legend_location)
    if output_path is not None:
        try:
            Writer.savePlot(output_path)
        except FileNotFoundError as error:
            logging.error(error)
    else:
        matplotlib.pyplot.show()
    