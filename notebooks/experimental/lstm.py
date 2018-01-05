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


def build(params):
    """
    Build the LSTM according to the parameters passed. The general
    architecture is set in the code.
    """
    model = Sequential()
    # Check if my design has more than 1 layer.
    ret_seq_flag = False
    if params['lstm_numlayers'] > 1:
        ret_seq_flag = True
    print('1st layer return sequence: {:s}'.format(str(ret_seq_flag)))
    # Add input layer.
    print('Adding layer #{:d} [{:d}]'
          .format(1, params['lstm_layer{:d}'.format(1)]))
    model.add(LSTM(
            params['lstm_layer1'],
            stateful=params['lstm_stateful'],
            unit_forget_bias=params['lstm_forget_bias'],
            unroll=params['lstm_unroll'],
            batch_input_shape=(params['lstm_batch_size'],
                               params['lstm_timesteps'],
                               params['num_features']),
            # input_shape=(params['lstm_timesteps'], params['num_features']),
            return_sequences=ret_seq_flag))
    model.add(Dropout(params['lstm_dropout1']))
    # Add additional hidden layers.
    for layer in range(1, params['lstm_numlayers']):
        if (layer+1) is params['lstm_numlayers']:
            ret_seq_flag = False
        print('Adding layer #{:d} [{:d}]'.format(
            layer+1, params['lstm_layer{:d}'.format(layer+1)]))
        model.add(LSTM(params['lstm_layer{:d}'.format(layer+1)],
                       return_sequences=ret_seq_flag))
        model.add(Dropout(params['lstm_dropout{:d}'.format(layer+1)]))

    # https://www.ijsr.net/archive/v6i4/ART20172755.pdf
    # model.add(Dense(8, kernel_initializer='uniform', activation='relu'))

    # Output layer.
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
