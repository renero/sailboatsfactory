from os.path import join
from os import getcwd
from pandas import read_csv, concat, DataFrame
from yaml import load

import matplotlib.pyplot as plt
import numpy as np

from math import sin
from math import pi
from math import exp
from random import randint
from random import uniform
from numpy import array, empty
from numpy.random import seed

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from tensorflow import set_random_seed


def plot_history(history):
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()


def read_parameters():
    """
    Reads a YAML file within the CWD of the current notebook to read all the params from there.
    """
    default_path = '/Users/renero/Documents/SideProjects/sailboatsfactory/notebooks/experimental'
    yaml_file = join(default_path, 'params.yaml')
    with open(yaml_file, 'r') as f:
        my_params = load(f)
    return my_params


def read_dataset(my_params):
    raw_dataset = read_csv(my_params['file_path'],
                           header='infer',
                           delimiter=my_params['delimiter'],
                           usecols=my_params['columNames'])
    # Remove the first column as it contains the value we want to predict
    # dataset = raw_dataset.iloc[:, 1:]
    my_params['raw_numrows'] = raw_dataset.shape[0]
    return(raw_dataset)


def prepare_data(raw, params):
    """
    Takes the data series as a sequence of rows, with "num_features" features on each line, and transform it into
    a 3D array, where the first dimension is the number of sliding windows formed (num_frames), of size
    "num_timesteps", each of them with "num_features" features.

    For num_timesteps=2, and 3 features on each row, the original
    raw data (5 x 3) is transformed into a (4 x 2 x 3)

    [1,2,3]     [[1,2,3],[1,4,5]]
    [1,4,5]  => [[1,4,5],[2,6,3]]
    [2,6,3]  => [[2,6,3],[4,2,3]]
    [4,2,3]     [[4,2,3],[5,3,8]]
    [5,3,8]

    The number of predictions is taken into account, as it limits the number of sliding windows.
    """
    # Setup the windowing of the dataset.
    num_samples = raw.shape[0]
    num_features = raw.shape[1]
    num_predictions = params['lstm_predictions']
    num_timesteps = params['lstm_timesteps']
    num_frames = num_samples - (num_timesteps + num_predictions) + 1
    # Update internal cache of parameters
    params['num_samples'] = num_samples
    params['num_features'] = num_features
    params['num_frames'] = num_frames

    dataset = raw[0:].astype('float32')
    scaled = scaler.fit_transform(dataset)
    df = DataFrame(data=scaled[:,:], index=range(0,scaled.shape[0]),
               columns=['var-{:d}'.format(i) for i in range(scaled.shape[1])])

    # Build the 3D array (num_frames, num_timesteps, num_features)
    X = empty((num_frames, num_timesteps, num_features))
    Y = empty((num_frames, num_predictions))
    for i in range(num_samples - (num_timesteps + num_predictions)):
        X[i] = df.values[i:i+num_timesteps,]
        Y[i] = df.values[i+num_timesteps:i+num_timesteps+num_predictions, 0]
    return X,Y

##################################################################################################

set_random_seed(2)
seed(2)
scaler = MinMaxScaler(feature_range=(-1, 1))
params = read_parameters()
raw = read_dataset(params)
X,Y = prepare_data(raw, params)

# Build the net.
model = Sequential()
model.add(LSTM(25, return_sequences=True, input_shape=(num_timesteps, num_features)))
model.add(LSTM(50))
model.add(Dense(num_predictions))
model.compile(loss='mae', optimizer='adam')

history = model.fit(X, Y, batch_size=5, epochs=20, validation_split=params['validation_ratio'])
plot_history(history)
