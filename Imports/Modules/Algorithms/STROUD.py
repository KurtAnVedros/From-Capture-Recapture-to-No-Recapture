import numpy
import logging

# DESCRIPTION: Given a test set this fucntion executes the STROUD method against a baseline to produce evaluation values (p-values)
# INPUTS:   (a) training_set - a baseline set
#           (b) test_set - the desired accuracy/information to be retained from the new transformed set
# RETURNS:  (a) test_p_values - a list holding the evaluation values
def executeSTROUD(training_set, test_set, reverse):

    logging.info("executeSTROUD: executing STROUD method")
    logging.debug("Argument training_set: %s", training_set)
    logging.debug("Argument test_set: %s", test_set)


    stroud = numpy.sort(training_set)
    if reverse in [True, "True", "true"]:
        stroud = numpy.flip(stroud, 0)
    test_p_values = (1 + stroud.shape[0] - numpy.searchsorted(stroud, test_set, side='right')) / (1 + stroud.shape[0])



    logging.debug("Argument test_p_values: %s", test_p_values)

    return test_p_values