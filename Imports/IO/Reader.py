import numpy
import os
import glob
#import boto
#from boto.s3.connection import OrdinaryCallingFormat
import boto3
import botocore
import json
#from sklearn.externals 
import joblib
import logging
import time
import pickle



# DESCRIPTION: Loads a signal stored on a .pfp1 file format from an S3 bucket into memory. The file is never copied to local filesystem.
# INPUTS: bucket_name - the name of the default bucket on s3
#         key - the name of the key to be read (it is in the form prefix/object_name)
# RETURNS: signal - an numpy array representation of a signal; typically a vector of 1xN dimensions where N ~=> 20,000
# NOTES: this function disregards the header of the file; in pfp1 file format the header is 50 bytes long
#TODO: add logging
#TODO: add error handling
def loadPFP1S3(bucket_name, key):

    s3 = boto3.resource("s3")
    bucket = s3.Bucket(bucket_name)
    s3_object = bucket.Object(key)
    content = s3_object.get()["Body"].read()
    if content:
        signal = numpy.frombuffer(content, dtype=numpy.float32, count=-1, offset=50)

    return signal


# DESCRIPTION: Loads a signal stored on a .pfp2 file format to disk into memory. The file is never copied to local filesystem.
# INPUTS: bucket_name - the name of the default bucket on s3
#         key - the name of the key to be read (it is in the form prefix/object_name)
# RETURNS: signal - an numpy array representation of a signal; typically a vector of 1xN dimensions where N ~=> 20,000
# NOTES: this function disregards the header of the file; in pfp2 file format the header is 56 bytes long
#TODO: add logging
#TODO: add error handling
def loadPFP2S3(bucket_name, key):
    s3 = boto3.resource("s3")
    bucket = s3.Bucket(bucket_name)
    s3_object = bucket.Object(key)
    content = s3_object.get()["Body"].read()
    if content:
        signal = numpy.frombuffer(content, dtype=numpy.float32, count=-1, offset=56)

    return signal



# DESCRIPTION:  Loads a number of signals from a 'folder' on S3 into memory as a numpy array.
#               Typically raw signals are included in different folders corresponding to different modes of operation.
#               This function can be used when one wants to load a specific mode to an nd-array and not the entire dataset.
# INPUTS:   (a) S3 bucket name
#           (b) prefix of the file to be read in S3
#           (b) type of the file, so far the values "pfp1" and "pfp2" are supported;
#           (c) label to be assigned to the signals in the folder
# RETURNS:  (a) data - numpy array of MxN dimensions where M is the number of signals in the folder;
#           (b) labels - numpy array of Mx1 dimensions where M is the number of signals in the folder.
# NOTES: The label values are assumed to be numerical. So far experiments only binary (values 0, 1) have been considered
#TODO: add error handling
def loadSubSetFromS3(bucket_name, prefix, file_type, label):

    logging.info("loadSubSetFromS3: Loading signals from S3 bucket")
    logging.debug("Argument bucket_name: %s", bucket_name)
    logging.debug("Argument prefix: %s", prefix)
    logging.debug("Argument file_type: %s", file_type)
    logging.debug("Argument label: %s", label)

    # retrieve the file names (keys) in the bucket prefix (folder)
    s3 = boto3.resource("s3")
    bucket = s3.Bucket(bucket_name)
    signals_list = []
    for s3_object in bucket.objects.filter(Prefix=prefix):
        key = s3_object.key
        if file_type == "pfp1":
            try:
                signals_list.append(loadPFP1S3(bucket_name, key))
            except FileNotFoundError as error:
                logging.error(error)
        elif file_type == "pfp2":
            try:
                signals_list.append(loadPFP1S3(bucket_name, key))
            except FileNotFoundError as error:
                logging.error(error)

    # transform the list to an ndarray for greater flexibility
    X = numpy.array(signals_list)

    # create the label for each signal
    length = X.shape[0]
    labels_list = [label] * length
    y = numpy.array(labels_list)

    logging.debug("Return data: %s", X)
    logging.debug("Return labels: %s", y)

    return X, y





# DESCRIPTION:  Loads a number of signals from a list of folders located on S3 into memory as a numpy array.
#               Typically raw signals are included in folders each corresponding to different modes of operation.
# INPUTS:   (a) bucket - the bucket in S3 containing the folders.
#           (b) prefix - a list containing the prefix names (the folder paths) in S3
#           (c) file_type_list - a list containing the types of the files contained in the n-th folder, so far the values "pfp1" and "pfp2" are supported;
#           (d) labels_list - a list containing the labels of the signals contained in the n-th folder
# RETURNS:  (a) numpy array of MxN dimensions where M is the number of signals in the folder;
#           (b) numpy array of Mx1 dimensions where M is the number of signals in the folder.
# NOTES: Typically, this function should be the first statement of any experiment.
#        This function supports reading from folders existing on the same bucket.
#TODO: add error handling
def loadDatasetFromS3(bucket_name, prefix_list, file_types_list, labels_list):

    logging.info("loadDatasetFromS3: Loading full dataset from a list of S3 folders")
    logging.debug("Argument bucket: %s", bucket_name)
    logging.debug("Argument prefix_list: %s", prefix_list)
    logging.debug("Argument file_types_list: %s", file_types_list)
    logging.debug("Argument file_types_list: %s", labels_list)

    X_temp = []
    y_temp = []

    #for each folder load and the signals contained and concatenate them to constract the final dataset
    for i, prefix in enumerate(prefix_list):
        data, labels = loadSubSetFromS3(bucket_name, prefix, file_types_list[i], labels_list[i])
        X_temp.extend(data)
        y_temp.extend(labels)

    X = numpy.array(X_temp)
    y = numpy.array(y_temp)

    logging.debug("Return value X dimensions: %s", X.shape)
    logging.debug("Return value X: %s", X)
    logging.debug("Return value y dimensions: %s", y.shape)
    logging.debug("Return value y: %s", y)

    return X, y






# DESCRIPTION: Loads a signal stored on a .pfp1 file format to disk into memory
# INPUTS: full path of the file in the local disk; the file must end in "/"
# RETURNS: signal - an numpy array representation of a signal; typically a vector of 1xN dimensions where N ~=> 20,000
# NOTES: this function disregards the header of the file; in pfp1 file format the header is 50 bytes long
def loadPFP1Local(file_path):

    logging.info("loadPFP1Local: Loading binary file (pfp1 format)")
    logging.debug("Argument file_path: %s", file_path)

    signal = []
    try:
        f = open(file_path, "rb")
        f.seek(50, os.SEEK_SET)
        signal = numpy.fromfile(f, dtype=numpy.float32)
        f.close()
    except FileNotFoundError as error:
        logging.error(error)

    logging.debug("Argument file_path: %s", signal)
    return signal

# DESCRIPTION: Loads a signal stored on a .pfp2 file format to disk into memory
# INPUTS: full path of the file in the local disk; the file must end in "/"
# RETURNS: signal - a numpy array representation of a signal; typically a vector of 1xN dimensions where N ~=> 20,000
# NOTES: this function disregards the header of the file; in pfp2 file format the header is 56 bytes long
def loadPFP2Local(file_path):

    logging.info("loadPFP2Local: Loading binary file (pfp2 format)")
    logging.debug("Argument file_path: %s", file_path)
    signal = []
    try:
        f = open(file_path, "rb")
        f.seek(56, os.SEEK_SET)
        signal = numpy.fromfile(f, dtype=numpy.float32)
        f.close()
    except FileNotFoundError as error:
        logging.error(error)

    logging.debug("Return signal: %s", signal)
    return signal







def loadSigMFDataLocal(file_path, start):

    logging.info("loadSigMFDataLocal: Loading data from binary file (SigMF format)")
    logging.debug("Argument file_path: %s", file_path)
    logging.debug("Argument start: %s", start)

    signal = []
    try:
        f = open(file_path, "rb")

        f.seek(start, os.SEEK_SET)
        signal = numpy.fromfile(f, dtype=numpy.float32)
        f.close()
    except FileNotFoundError as error:
        logging.error(error)

    logging.debug("Return signal: %s", signal)

    return signal



def loadSubset(paths, label):


    signals_list = []
    for file in paths:
        try:
            signal = loadSigMFDataLocal(file, 0)
            signals_list.append(signal)
        except FileNotFoundError as error:
            print(error)

    labels = numpy.array([label] * len(signals_list))
    signals = numpy.array(signals_list)

    return signals, labels



# DESCRIPTION:  Loads a number of signals from a folder on the disk into memory as a numpy array.
#               Typically raw signals are included in different folders corresponding to different modes of operation.
# INPUTS:   (a) full path of the folder in the local disk;
#           (b) type of the file, so far the values "pfp1" and "pfp2" are supported;
#           (c) label to be assigned to the signals in the folder
# RETURNS:  (a) data - numpy array of MxN dimensions where M is the number of signals in the folder;
#           (b) labels - numpy array of Mx1 dimensions where M is the number of signals in the folder.
# NOTES: The label values are assumed to be numerical. So far experiments only binary (values 0, 1) have been considered
def loadSubSetFromLocalFolder(folder_path, file_type, label):

    logging.info("loadSubSetFromLocalFolder: Loading signals from local folder")
    logging.debug("Argument folder_path: %s", folder_path)
    logging.debug("Argument file_type: %s", file_type)
    logging.debug("Argument label: %s", label)

    #retrieve the file names in the folder
    try:
        files_list = glob.glob(folder_path + "*." + file_type)
    except FileNotFoundError as error:
        logging.error(error)

    #create a list of ndarrays (each array corresponds to a signal)
    #a list structure is preferred over an ndarray due to the continous concatenations (is more efficient)
    signals_list = []
    for file in files_list:
        #TODO: add unrecognized file extension error
        if file_type == "pfp1":
            try:
                signals_list.append(loadPFP1Local(file))
            except FileNotFoundError as error:
                logging.error(error)
        elif file_type == "pfp2":
            try:
                signals_list.append(loadPFP2Local(file))
            except FileNotFoundError as error:
                logging.error(error)

    #transform the list to an ndarray for greater flexibility
    data = numpy.array(signals_list)

    #create the label for each signal
    length = data.shape[0]
    labels_list = [label] * length
    labels = numpy.array(labels_list)

    logging.debug("Return data: %s", data)
    logging.debug("Return labels: %s", labels)

    return data, labels


# DESCRIPTION:  Loads a number of signals from a folder on the disk into memory as a numpy array.
#               Typically raw signals are included in different folders corresponding to different modes of operation.
# INPUTS:   (a) full path of the folder in the local disk;
#           (b) type of the file, so far the values "pfp1" and "pfp2" are supported;
#           (c) label to be assigned to the signals in the folder
# RETURNS:  (a) numpy array of MxN dimensions where M is the number of signals in the folder;
#           (b) numpy array of Mx1 dimensions where M is the number of signals in the folder.
# NOTES: Typically, this function should be the first statement of any experiment
def loadDatasetFromLocalFolders(folder_paths_list, file_types_list, labels_list):

    logging.info("loadDatasetFromLocalFolders: Loading dataset from a list of local folders")
    logging.debug("Argument folder_paths_list: %s", folder_paths_list)
    logging.debug("Argument file_types_list: %s", file_types_list)
    logging.debug("Argument file_types_list: %s", labels_list)

    X_temp = []
    y_temp = []

    #for each folder load and the signals contained and concatenate them to constract the final dataset
    for i, folder in enumerate(folder_paths_list):
        data, labels = loadSubSetFromLocalFolder(folder, file_types_list[i], labels_list[i])
        X_temp.extend(data)
        y_temp.extend(labels)

    X = numpy.array(X_temp)
    y = numpy.array(y_temp)

    logging.debug("Return value X dimensions: %s", X.shape)
    logging.debug("Return value X: %s", X)
    logging.debug("Return value y dimensions: %s", y.shape)
    logging.debug("Return value y: %s", y)

    return X, y


#TODO: do some experiments and elaborate on the description
# DESCRIPTION:  Loads a stored model from the disk to memory.
#
# INPUTS:   (a) model_file_path - full path of the binary that holds the model;
# RETURNS:  (a) model - a model object
# NOTES:
def loadModelFromLocalDisk(model_file_path):

    logging.info("loadModel: Loading a model")
    logging.debug("Argument model_file_path: %s", model_file_path)
    try:
        model = joblib.load(model_file_path)
    except FileNotFoundError as error:
        logging.error(error)



    logging.debug("Return value model: %s", str(model))

    return model


# DESCRIPTION:  Loads a stored model from the S3 to memory.
# INPUTS:   (a) bucket_name - the bucket name where the model file exists on S3
#           (b) key - the key of the file to be loaded in the S3 folder
#           (c) temp_output_path - the path on the local hard drive where the temporary file will be saved
# RETURNS:  (a) model - a model object
# NOTES:    The model will first be copied on the local drive, fully loaded into memory and then be deleted
def loadModelFromS3(bucket_name, key, temp_output_path):

    logging.info("loadModelFromS3: Loading a model")
    logging.debug("Argument bucket_name: %s", bucket_name)
    logging.debug("Argument key: %s", key)
    logging.debug("Argument temp_output_path: %s", temp_output_path)

    try:
        #load the model from S3
        s3 = boto3.resource("s3")
        try:
            s3.Bucket(bucket_name).download_file(key, temp_output_path)
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == "404":
                print("The object does not exist.")
            else:
                raise
        #save the model to local drive, to open it with joblib
        model = joblib.load(temp_output_path)
        #open the model with joblib
        os.remove(temp_output_path)
    except FileNotFoundError as error:
        logging.error(error)

    logging.debug("Return value model: %s", str(model))

    return model



# TODO: possible source of bugs load function does not have arguments for defining the type
# DESCRIPTION:  Loads a dataset located in local drive stored in .npy binary format using the numpy's saveDataSet function
#
# INPUTS:   (a) file_path - full path of the binary that holds the dataset;
# RETURNS:  (a) dataset - the dataset in numpy array format
# NOTES: the dataset read, must have been first saved as a .npy file
def loadDatasetFromBinaryFileLocal(file_path):

    logging.info("loadDatasetFromBinaryFileLocal: Loading dataset from disk")
    logging.debug("Argument file_path: %s", file_path)
    #dataset = numpy.fromfile(file_path, dtype=numpy.float32)
    try:
        dataset = numpy.load(file_path)
    except FileNotFoundError as error:
        logging.error(error)

    logging.debug("Return value dataset: %s", dataset)

    return dataset


# DESCRIPTION:  Loads a dataset located in local drive stored in raw binary format
#
# INPUTS:   (a) file_path - full path of the binary that holds the dataset;
#           (b) rows - the number of rows the read dataset will contain;
#           (c) columns - the number of columns the read dataset will contain;
# RETURNS:  (a) dataset - the dataset in numpy array format
# NOTES: the dataset read, must have been first saved as a raw binary file (e.g., .bin or .data format);
#       if the dataset was stored as a .npy binary file then the
def loadDatasetFromRawBinaryFileLocal(file_path, rows, columns):

    logging.info("loadDatasetFromRawBinaryFileLocal: Loading dataset from disk")
    logging.debug("Argument file_path: %s", file_path, rows, columns)
    try:
        data = numpy.fromfile(file_path, dtype=numpy.float64)
        temp = numpy.reshape(data, (columns, rows))
        dataset = numpy.transpose(temp)
    except FileNotFoundError as error:
        logging.error(error)

    logging.debug("Return value dataset: %s", dataset)

    return dataset



# TODO: possible source of bugs load function does not have arguments for defining the type
# DESCRIPTION:  Loads a dataset located in S3 in binary format using the saveDataSet function
#
# INPUTS:   (a) bucket_name - full path of the binary that holds the dataset;
#           (b) key - key of the file in the S3
#           (c) temp_output_path -  the path on the local hard drive where the temporary file will be saved
# RETURNS:  (a) dataset - the dataset in numpy array format
# NOTES:    the dataset read, must have been first saved as a .npy file
def loadDatasetFromBinaryFileS3(bucket_name, key, temp_output_path):

    logging.info("loadDatasetFromBinaryFileS3: Loading dataset from disk")
    logging.debug("Argument bucket_name: %s", bucket_name)
    logging.debug("Argument key: %s", key)
    logging.debug("Argument temp_output_path: %s", temp_output_path)
    #dataset = numpy.fromfile(file_path, dtype=numpy.float32)

    s3 = boto3.resource("s3")
    try:
        s3.Bucket(bucket_name).download_file(key, temp_output_path)
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            print("The object does not exist.")
        else:
            raise
    try:
        dataset = numpy.load(temp_output_path)
        os.remove(temp_output_path)
    except FileNotFoundError as error:
        logging.error(error)

    logging.debug("Return value dataset: %s", dataset)

    return dataset


# DESCRIPTION:  Loads a list stored in binary format using the saveList function.
#
# INPUTS:   (a) file_path - full path of the binary that holds the list;
# RETURNS:  (a) list - stored list
# NOTES: -
def loadListLocal(file_path):

    logging.info("loadList: Loading list from disk")
    logging.debug("Argument file_path: %s", file_path)

    try:
        with open(file_path, "rb") as fp:
            list = pickle.load(fp)
    except FileNotFoundError as error:
        logging.error(error)

    logging.debug("Return value list: %s", list)

    return list


# DESCRIPTION:  Loads a list stored that was previously stored in binary format (using the saveList function).
#
# INPUTS:   (a) bucket_name - full path of the binary that holds the list;
#           (b) key - key of the file in the S3
#           (c) temp_output_path - the path on the local hard drive where the temporary file will be saved
# RETURNS:  (a) list - stored list
# NOTES: this function will first download the file on the local drive, load it and then delete it
def loadListS3(bucket_name, key, temp_output_path):

    logging.info("loadListS3: Loading list from disk")
    logging.debug("Argument bucket_name: %s", bucket_name)
    logging.debug("Argument key: %s", key)
    logging.debug("Argument temp_output_path: %s", temp_output_path)

    s3 = boto3.resource("s3")
    try:
        s3.Bucket(bucket_name).download_file(key, temp_output_path)
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            print("The object does not exist.")
        else:
            raise

    try:
        with open(temp_output_path, "rb") as fp:
            list = pickle.load(fp)
            os.remove(temp_output_path)
    except FileNotFoundError as error:
        logging.error(error)

    logging.debug("Return value list: %s", list)

    return list


# DESCRIPTION:  Loads a json file -typically containing parameters- into memory
#
# INPUTS:   (a) file_path - full path of the json file;
# RETURNS:  (a) json_object - a dictionary repsresenting the json object
# NOTES: -
def loadJSONLocal(file_path):

    try:
        with open(file_path) as json_data:
            json_object = json.load(json_data)
    except FileNotFoundError as error:
        logging.error(error)

    return json_object



# DESCRIPTION:  Loads a json file -typically containing parameters- into memory
#
# INPUTS:   (a) file_path - full path of the json file;
# RETURNS:  (a) json_object - a dictionary repsresenting the json object
# NOTES: -
def loadJSONS3(bucket_name, key, temp_output_path):

    s3 = boto3.resource("s3")
    try:
        s3.Bucket(bucket_name).download_file(key, temp_output_path)
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            print("The object does not exist.")
        else:
            raise

    try:
        with open(temp_output_path) as json_data:
            json_object = json.load(json_data)
    except FileNotFoundError as error:
        logging.error(error)

    return json_object
