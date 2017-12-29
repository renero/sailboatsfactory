from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import numpy as np
from numpy import array, empty
from numpy.random import seed
from os.path import join
from os import getcwd
from pandas import read_csv, concat, DataFrame
from random import randint
from random import uniform
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow import set_random_seed
from yaml import load

# Initialization of seeds
set_random_seed(2)
seed(2)

def parameters():
    """
    Reads a YAML file within the CWD of the current notebook to read all the
    params from there.
    """
    default_path = '/Users/renero/Documents/SideProjects/sailboatsfactory/notebooks/experimental'
    yaml_file = join(default_path, 'params.yaml')
    with open(yaml_file, 'r') as f:
        my_params = load(f)
    my_params['x_scaler'] = MinMaxScaler(feature_range=(-1, 1))
    my_params['y_scaler'] = MinMaxScaler(feature_range=(-1, 1))
    return my_params


def build(params):
    """
    Build the LSTM according to the parameters passed. The general
    architecture is set in the code.
    """
    model = Sequential()
    model.add(
        LSTM(
            params['lstm_1stlayer'],
            input_shape=(params['lstm_timesteps'], params['num_features']),
            return_sequences=True))
    model.add(Dropout(params['lstm_dropout1']))
    model.add(LSTM(params['lstm_2ndlayer']))
    model.add(Dropout(params['lstm_dropout2']))
    # https://www.ijsr.net/archive/v6i4/ART20172755.pdf
    # model.add(Dense(16, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(params['lstm_predictions']))
    model.compile(loss=params['lstm_loss'], optimizer=params['lstm_optimizer'])
    return model


def predict(model, test_X, my_params, invert=True):
    """
    Make a prediction with the model over the test_X dataset as input.
    """
    yhat = model.predict(test_X, batch_size=my_params['lstm_batch_size'])
    if invert is False:
        return yhat
    inv_yhat = invert_Y(test_X, yhat, my_params)
    return inv_yhat
