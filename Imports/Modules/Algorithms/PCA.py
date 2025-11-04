from sklearn.decomposition import PCA
from IO import Writer
import logging



# DESCRIPTION:  Executes Particle Component Analysis to produce dataset with slightly less information but significantly less number of attributes
# INPUTS:   (a) training_set - the set upon which the transformation will be applied without the labels
#           (b) test_set - the set of signals that is tested
#           (c) number_components - the number of desired components (features) to be produced in the transformed set, typically significantly less than the features of the original set
#           (d) save_model - the algorithm used for estimating the distance between two signals; by default the euclidean distance
#           (e) model_path - the file path for the model to be saved
#           (f) save_transformed_training_set - a boolean indicating whether the transformed dataset is to be saved to disk
#           (g) output_training_set_path - the file path where the transformed training set is to be saved to disk
#           (h) save_transformed_test_set
#           (i) output_test_set_path - the file path where the transformed test set is to be saved to disk
# RETURNS:  (a) transformed_training_set - a transformed -according to the PCA- training set
#           (b) transformed_test_set - a transformed -according to the PCA- test set
# NOTES:
def executePCA(training_set, test_set, number_components=None, save_model=False, model_path=None, output_training_set_path=None, output_test_set_path=None):

    logging.info("executePCA: Executing PCA")
    logging.debug("Argument training_set: %s", training_set)
    logging.debug("Argument test_set: %s", test_set)
    logging.debug("Argument number_components: %s", number_components)
    logging.debug("Argument save_model: %s", save_model)
    logging.debug("Argument model_path: %s", model_path)
    logging.debug("Argument output_training_set_path: %s", output_training_set_path)
    logging.debug("Argument output_test_set_path: %s", output_test_set_path)

    model = PCA(number_components)
    model.fit(training_set)

    if save_model is True:
        try:
            Writer.saveModel(model, model_path)
        except FileNotFoundError as error:
            logging.debug(error)

    transformed_training_set = model.transform(training_set)

    if output_training_set_path is not None:
        try:
            Writer.saveDataSet(transformed_training_set, output_training_set_path)
        except FileNotFoundError as error:
            logging.debug(error)

    transformed_test_set = model.transform(test_set)

    if output_test_set_path is not None:
        try:
            Writer.saveDataSet(transformed_test_set, output_test_set_path)
        except FileNotFoundError as error:
            logging.debug(error)

    logging.debug("Return value transformed_training_set: %s", transformed_training_set)
    logging.debug("Return value transformed_test_set: %s", transformed_test_set)

    return transformed_training_set, transformed_test_set


# DESCRIPTION:  Calculates the number of components that need to be produced to achieve a desired ratio of information retained
# INPUTS:   (a) data - the original dataset
#           (b) desired_accuracy - the desired accuracy/information to be retained from the new transformed set
# RETURNS:  (a) transformed_training_set - a transformed -according to the PCA- training set
#           (b) transformed_test_set - a transformed -according to the PCA- test set
# NOTES: This function goes through the process of fully transforming the dataset.
# Each component (new feature) retains/expresses of ratio information from the overall dataset
# This is a possible point of optimization in the future
def getNumberOfComponentsToAchieveRatio(data, desired_accuracy):

    logging.info("getNumberOfComponentsToAchieveRatio: calculating the number of components to be produced to achieve the desired information ratio")
    logging.debug("Argument data: %s", data)
    logging.debug("Argument desired_accuracy: %s", desired_accuracy)

    model = PCA()
    model.fit(data)
    results = model.transform(data)

    cummulated = model.explained_variance_ratio_.cumsum()

    for index, item in enumerate(cummulated):
        if item >= desired_accuracy:
            logging.debug("Return value index: %s", index)
            return index

    logging.debug("Return value index: %s", -1)
    return -1
