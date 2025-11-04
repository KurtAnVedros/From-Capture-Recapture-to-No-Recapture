#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt

from sklearn.preprocessing import StandardScaler


# In[7]:


# DESCRIPTION:  creates the sequence dataset
# INPUTS:   (a) x - the training signal
#           (b) y - the signal to test
# RETURNS:  (a) base_lof - the lof for the base set
#           (b) test_lof - the lof for the test set
def create_sequences(values, time_steps=10):
    output = []
    for i in range(len(values) - time_steps):
        output.append(values[i : (i + time_steps)])
    return np.stack(output)

def normalize_test(values, mean, std):
    values -= mean
    values /= std
    return values

def plotTimeSeries(x, y, title):
    plt.plot(x, y)
    axes = plt.gca()
    axes.set_title(title)
    axes.set_xlabel('Sample Index')
    axes.set_ylabel('Amplitude')
    plt.show()
    
def plotTimeSeriesOverlay(x, y, index, title):
    
    colors =[]
    for i in range(0, len(y)):
        if i in index:
            colors.append("red")
        else:
            colors.append("blue")
            
    plt.plot(x, y, color = colors)
    axes = plt.gca()
    axes.set_title(title)
    axes.set_xlabel('Sample Index')
    axes.set_ylabel('Amplitude')
    plt.show()


# # test

# In[8]:


# DESCRIPTION:  Executes (a modified version of the) local outlier function to measure the strangeness between signals in two sets
# INPUTS:   (a) x - the training signal
#           (b) y - the signal to test
# RETURNS:  (a) base_lof - the lof for the base set
#           (b) test_lof - the lof for the test set
def autoencoderTest(x, y, TIME_STEPS):  
    
    train = x.flatten() 
    test = y
    
    timeTrain = list(range(0, len(train)))
    timeTest = list(range(0, len(test)))
    
    print("Train shape: ", train.shape)
    print("Test shape: ", test.shape)
    
    train = train[:,np.newaxis]
    test = test[:,np.newaxis]
    
    print("Train shape: ", train.shape)
    print("Test shape: ", test.shape)
    
    # scale the two signals to the same length.

    # reshape to [samples, time_steps, n_features]
    training_mean = train.mean()
    training_std = train.std()
    training_value = (train - training_mean) / training_std
    print("Number of training samples:", len(training_value))

    x_train = create_sequences(training_value, TIME_STEPS)
    print("Training input shape: ", x_train.shape)
    
    # create the model
    model = keras.Sequential(
        [
            layers.Input(shape=(x_train.shape[1], x_train.shape[2])),
            layers.Conv1D(
                filters=32, kernel_size=7, padding="same", strides=2, activation="relu"
            ),
            layers.Dropout(rate=0.2),
            layers.Conv1D(
                filters=16, kernel_size=7, padding="same", strides=2, activation="relu"
            ),
            layers.Conv1DTranspose(
                filters=16, kernel_size=7, padding="same", strides=2, activation="relu"
            ),
            layers.Dropout(rate=0.2),
            layers.Conv1DTranspose(
                filters=32, kernel_size=7, padding="same", strides=2, activation="relu"
            ),
            layers.Conv1DTranspose(filters=1, kernel_size=7, padding="same"),
        ]
    )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")

    print(model.summary())
    
    # train
    history = model.fit(
        x_train,
        x_train,
        epochs=50,
        batch_size=128,
        validation_split=0.1,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min")
        ],
    )

    # show training and validation loss
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.legend()
    plt.show()
    
    # Detect anomalies
    # Get train MAE loss.
    x_train_pred = model.predict(x_train)
    train_mae_loss = np.mean(np.abs(x_train_pred - x_train), axis=1)

    plt.hist(train_mae_loss, bins=50)
    plt.xlabel("Train MAE loss")
    plt.ylabel("No of samples")
    plt.show()

    # Get reconstruction loss threshold.
    threshold = np.max(train_mae_loss)
    print("Reconstruction error threshold: ", threshold)

    # Checking how the first sequence is learnt
    print("Check how the first sequence is learned")
    plt.plot(x_train[0])
    plt.plot(x_train_pred[0])
    plt.show()

    # Prepare test data
    test_value = (test - training_mean) / training_std
    plotTimeSeries(timeTest, test_value, "Test Value")

    # Create sequences from test values.
    x_test = create_sequences(test_value, TIME_STEPS)
    print("Test input shape: ", x_test.shape)

    # Get test MAE loss.
    x_test_pred = model.predict(x_test)
    test_mae_loss = np.mean(np.abs(x_test_pred - x_test), axis=1)
    test_mae_loss = test_mae_loss.reshape((-1))

    plt.hist(test_mae_loss, bins=50)
    plt.xlabel("test MAE loss")
    plt.ylabel("No of samples")
    plt.show()
    
    # Check how the first sequence was learned
    print("Check how the first sequence is learned")
    plt.plot(x_test[0])
    plt.plot(x_test_pred[0])
    plt.show()

    # Detect all the samples which are anomalies.
    anomalies = test_mae_loss > threshold
    print("Number of anomaly samples: ", np.sum(anomalies))
    print("Indices of anomaly samples: ", np.where(anomalies))

    # plot anomalies
    # data i is an anomaly if samples [(i - timesteps + 1) to (i)] are anomalies
    anomalous_data_indices = []
    for data_idx in range(TIME_STEPS - 1, len(test_value) - TIME_STEPS + 1):
        if np.all(anomalies[data_idx - TIME_STEPS + 1 : data_idx]):
            anomalous_data_indices.append(data_idx)

    subset = test_value[anomalous_data_indices]
    print(subset)
    time = list(range(0, len(subset)))
    plotTimeSeries(time, subset, "test")


# In[ ]:


# DESCRIPTION:  Executes (a modified version of the) local outlier function to measure the strangeness between signals in two sets
# INPUTS:   (a) x - the training signal
#           (b) y - the signal to test
# RETURNS:  (a) base_lof - the lof for the base set
#           (b) test_lof - the lof for the test set
def autoencoderTest2(x, y, TIME_STEPS):  
    train = x
    test = y
    
    timeTrain = list(range(0, len(train)))
    timeTest = list(range(0, len(test)))
    
    print("Train shape: ", train.shape)
    print("Test shape: ", test.shape)
    
    train = train[:,np.newaxis]
    test = test[:,np.newaxis]
    
    print("Train shape: ", train.shape)
    print("Test shape: ", test.shape)
    
    # scale the two signals to the same length.

    # reshape to [samples, time_steps, n_features]
    training_mean = train.mean()
    training_std = train.std()
    training_value = (train - training_mean) / training_std
    print("Number of training samples:", len(training_value))

    #x_train = create_sequences(training_value, TIME_STEPS)
    #print("Training input shape: ", x_train.shape)
    x_train = train
    
    # create the model
    model = keras.Sequential(
        [
            layers.Input(shape=(x_train.shape[0], x_train.shape[1])),
            layers.Conv1D(
                filters=32, kernel_size=7, padding="same", strides=2, activation="relu"
            ),
            layers.Dropout(rate=0.2),
            layers.Conv1D(
                filters=16, kernel_size=7, padding="same", strides=2, activation="relu"
            ),
            layers.Conv1DTranspose(
                filters=16, kernel_size=7, padding="same", strides=2, activation="relu"
            ),
            layers.Dropout(rate=0.2),
            layers.Conv1DTranspose(
                filters=32, kernel_size=7, padding="same", strides=2, activation="relu"
            ),
            layers.Conv1DTranspose(filters=1, kernel_size=7, padding="same"),
        ]
    )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")

    print(model.summary())
    
    # train
    history = model.fit(
        x_train,
        x_train,
        epochs=50,
        batch_size=128,
        validation_split=0.1,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min")
        ],
    )

    # show training and validation loss
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.legend()
    plt.show()
    
    # Detect anomalies
    # Get train MAE loss.
    x_train_pred = model.predict(x_train)
    train_mae_loss = np.mean(np.abs(x_train_pred - x_train), axis=1)

    plt.hist(train_mae_loss, bins=50)
    plt.xlabel("Train MAE loss")
    plt.ylabel("No of samples")
    plt.show()

    # Get reconstruction loss threshold.
    threshold = np.max(train_mae_loss)
    print("Reconstruction error threshold: ", threshold)

    # Checking how the first sequence is learnt
    plt.plot(x_train[0])
    plt.plot(x_train_pred[0])
    plt.show()

    # Prepare test data
    test_value = (test - training_mean) / training_std
    plotTimeSeries(timeTest, test_value, "Test Value")

    # Create sequences from test values.
    x_test = create_sequences(test_value, TIME_STEPS)
    print("Test input shape: ", x_test.shape)

    # Get test MAE loss.
    x_test_pred = model.predict(x_test)
    test_mae_loss = np.mean(np.abs(x_test_pred - x_test), axis=1)
    test_mae_loss = test_mae_loss.reshape((-1))

    plt.hist(test_mae_loss, bins=50)
    plt.xlabel("test MAE loss")
    plt.ylabel("No of samples")
    plt.show()

    # Detect all the samples which are anomalies.
    anomalies = test_mae_loss > threshold
    print("Number of anomaly samples: ", np.sum(anomalies))
    print("Indices of anomaly samples: ", np.where(anomalies))

    # plot anomalies
    # data i is an anomaly if samples [(i - timesteps + 1) to (i)] are anomalies
    anomalous_data_indices = []
    for data_idx in range(TIME_STEPS - 1, len(test_value) - TIME_STEPS + 1):
        if np.all(anomalies[data_idx - TIME_STEPS + 1 : data_idx]):
            anomalous_data_indices.append(data_idx)

    subset = test_value[anomalous_data_indices]
    print(subset)
    time = list(range(0, len(subset)))
    plotTimeSeries(time, subset, "test")

