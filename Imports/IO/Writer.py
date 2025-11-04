#from sklearn.externals import joblib
import joblib
import numpy
import matplotlib.pyplot
import pickle
import logging
import json
#import boto3
import os


# DESCRIPTION: Saves a model trained with an algorithm to disk
# INPUTS: (a) model - the model produced by the algorithm
#         (b) file_path - the output file
# RETURNS: -
# NOTES: -
def saveModel(model, file_path, bucket_name=None, key=None, binary_data=None):

    logging.info("saveModel: Saving model to disk")
    logging.debug("Argument model: %s", str(model))
    logging.debug("Argument file_path: %s", file_path)

    try:
        directory = os.path.dirname(file_path)
        os.makedirs(directory, exist_ok=True)
        joblib.dump(model, file_path)
    except IOError as error:
        logging.error(error)

    if bucket_name is not None and key is not None and binary_data is not None:
        binary_data = open(file_path, 'rb')
        saveBinaryFileS3(bucket_name, key, binary_data)
        os.remove(file_path)

# DESCRIPTION: Saves a dataset -represented as numpy array-, into the disk
# INPUTS: (a) dataset - the dataset to be saved
#         (b) file_path - the output file
# RETURNS: -
# NOTES: It is usually used to store datasets after transformation to disk.
#        In unsupervised learning most implemented approaches have their models as numpy arrays.
#        Therefore, this methods is used
def saveDataSet(dataset, file_path, bucket_name=None, key=None, binary_data=None):

    logging.info("saveDataSet: Saving dataset to disk")
    logging.debug("Argument dataset: %s", dataset)
    logging.debug("Argument file_path: %s", file_path)

    try:
        directory = os.path.dirname(file_path)
        os.makedirs(directory, exist_ok=True)
        numpy.save(file_path, dataset)
    except IOError as error:
        logging.error(error)

    if bucket_name is not None and key is not None and binary_data is not None:
        binary_data = open(file_path, 'rb')
        saveBinaryFileS3(bucket_name, key, binary_data)
        os.remove(file_path)







# DESCRIPTION: Save a dictionary to disk
# INPUTS: (a) dictionary - the dictionary to be saved
#         (b) file_path - the output file
# RETURNS: -
# NOTES: It is usually used to store results or intermediate data e.g., the
def saveDictionary(dictionary, file_path, bucket_name=None, key=None, binary_data=None):

    logging.info("saveDictionary: Saving dictionary to disk")
    logging.debug("Argument dictionary: %s", dictionary)
    logging.debug("Argument file_path: %s", file_path)

    try:
        directory = os.path.dirname(file_path)
        os.makedirs(directory, exist_ok=True)
        with open(file_path, "w") as fp:
            #pickle.dump(list, fp)
            json.dump(dictionary, fp)
    except IOError as error:
        logging.error(error)

    if bucket_name is not None and key is not None and binary_data is not None:
        binary_data = open(file_path, 'rb')
        saveBinaryFileS3(bucket_name, key, binary_data)
        os.remove(file_path)


# DESCRIPTION: Save a list to disk
# INPUTS: (a) list - the list to be saved
#         (b) file_path - the output file
# RETURNS: -
# NOTES: It is usually used to store results or intermediate data after some processing to disk
def saveList(list, file_path, bucket_name=None, key=None, binary_data=None):

    logging.info("saveList: Saving list to disk")
    logging.debug("Argument list: %s", list)
    logging.debug("Argument file_path: %s", file_path)

    try:
        directory = os.path.dirname(file_path)
        os.makedirs(directory, exist_ok=True)
        with open(file_path, "wb") as fp:
            pickle.dump(list, fp)
    except IOError as error:
        logging.error(error)

    if bucket_name is not None and key is not None and binary_data is not None:
        binary_data = open(file_path, 'rb')
        saveBinaryFileS3(bucket_name, key, binary_data)
        os.remove(file_path)


# DESCRIPTION: Save a plot to disk
# INPUTS: (a) file_path - the output file
# RETURNS: -
# NOTES: -
def savePlot(file_path, bucket_name=None, key=None, binary_data=None):

    logging.info("savePlot: Saving plot to disk")
    logging.debug("Argument file_path: %s", file_path)

    try:
        directory = os.path.dirname(file_path)
        os.makedirs(directory, exist_ok=True)
        matplotlib.pyplot.savefig(file_path, bbox_inches="tight")
        matplotlib.pyplot.close()
    except IOError as error:
        logging.error(error)

    if bucket_name is not None and key is not None and binary_data is not None:
        binary_data = open(file_path, 'rb')
        saveBinaryFileS3(bucket_name, key, binary_data)
        os.remove(file_path)

# DESCRIPTION: Saves binary data to an S3 location. The binary data are sourced from a file on the local disk
# INPUTS: (a) bucket_name - the name of the bucket in S3
#         (b) key - the key in the S3 under which the binary data will be stored
#         (c) binary_data - that binary data to be stored in S3
# RETURNS: -
# NOTES: Typically the file in S3 must have a specific structure therefore it is desirable to have first saved the data in the appropriate format and then load it.
def saveBinaryFileS3(bucket_name, key, binary_data):
    s3 = boto3.resource('s3')
    s3_object = s3.Object(bucket_name, key)
    s3_object.put(Body=binary_data)




# DESCRIPTION: Saves a dataset -represented as numpy array-, into the disk in a raw binary format
# INPUTS: (a) dataset - the dataset to be saved
#         (b) file_path - the output file
# RETURNS: -
# NOTES: It is usually used to store datasets after transformation to disk.
#        In unsupervised learning most implemented approaches have their models as numpy arrays.
#        Unlike saveDataSet this method save the data in raw binary format and not npy format.
def saveDataSetRawBinary(dataset, file_path):

    logging.info("saveDataSetRawBinary: Saving dataset to disk")
    logging.debug("Argument dataset: %s", dataset)
    logging.debug("Argument file_path: %s", file_path)

    try:
        directory = os.path.dirname(file_path)
        os.makedirs(directory, exist_ok=True)
        dataset.tofile(file_path)
    except IOError as error:
        logging.error(error)
