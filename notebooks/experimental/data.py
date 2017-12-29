from os.path import join
from os import getcwd
from pandas import read_csv, concat, DataFrame
from yaml import load

import numpy as np

from random import randint
from random import uniform
from numpy import array, empty
from numpy.random import seed


def read(my_params):
    raw_dataset = read_csv(
        my_params['file_path'],
        header='infer',
        delimiter=my_params['delimiter'],
        usecols=my_params['columNames'])
    # Remove the first column as it contains the value we want to predict
    # dataset = raw_dataset.iloc[:, 1:]
    my_params['raw_numrows'] = raw_dataset.shape[0]
    return (raw_dataset)


def diff(a, interval=1):
    """
    Given a 2D array (a), compute the one resulting from differentiating each
    element by the one at "interval" distance from it, columwise:

     [  2.   4.   6.]  -> [  0.  40.   2.]
     [  2.  44.   8.]  -> [  1. -39.  -2.]
     [  3.   5.   6.]  -> [  0.   7.   8.]
     [  3.  12.  14.]  -> [  2.  76.  -8.]
     [  5.  88.   6.]

    """
    if a.ndim is not 2:
        raise ValueError(
            'Differentiating tensor with wrong number of dimensions ({:d})'.
            format(a.ndim))
    b = np.empty(((a.shape[0] - interval), a.shape[1]))
    for row in range(interval, len(a)):
        for col in range(a.shape[1]):
            b[row - interval][col] = a[row][col] - a[row - interval][col]
    return b


def inv_diff(a, b, interval=1):
    """
    Inverts the operation at 'diff', needing the original vector.
    """
    if a.ndim is not 2:
        raise ValueError(
            'Differentiating with wrong number of dimensions({:d})'.format(
                a.ndim))
    c = np.empty((b.shape[0], b.shape[1]))
    for col in range(b.shape[1]):
        c[:, col] = np.concatenate(
            (b[0:interval, col], a[:, col] + b[:-interval, col]), axis=0)
    return c


def split(X, Y, num_testcases):
    return X[:-num_testcases, ], Y[:-num_testcases, ], X[-num_testcases:, ], Y[-num_testcases:, ]


def prepare(raw, params):
    """
    Takes the data series as a sequence of rows, with "num_features" features
    on each line, and transform it into a 3D array, where the first dimension
    is the number of sliding windows formed (num_frames), of size
    "num_timesteps", each of them with "num_features" features.

    For num_timesteps=2, 3 features on each row, and making 1 prediction in
    the future, the original raw data (5 x 3) is transformed into:
    --> (3 x (2+1) x 3) = (num_frames x (num_timesteps + num_predictions) x num_features):

    [1,2,3]     ------- X ------ -- Y --
    [1,4,5]     [[1,2,3],[1,4,5],[2,_,_]]
    [2,6,3]  => [[1,4,5],[2,6,3],[4,_,_]]
    [4,2,3]     [[2,6,3],[4,2,3],[5,_,_]]
    [5,3,8]
    """
    # Diff, first of all.
    non_stationary = np.array((diff(raw.values)))
    # Setup the windowing of the dataset.
    num_samples = non_stationary.shape[0]
    num_features = non_stationary.shape[1]
    num_predictions = params['lstm_predictions']
    num_timesteps = params['lstm_timesteps']
    num_frames = num_samples - (num_timesteps + num_predictions) + 1
    # Update internal cache of parameters
    params['num_samples'] = num_samples
    params['num_features'] = num_features
    params['num_frames'] = num_frames

    # Build the 3D array (num_frames, num_timesteps, num_features)
    X = empty((num_frames, num_timesteps, num_features))
    Y = empty((num_frames, num_predictions))
    for i in range(num_samples - (num_timesteps + num_predictions)):
        X[i] = non_stationary[i:i + num_timesteps, ]
        Y[i] = non_stationary[i + num_timesteps:i + num_timesteps + num_predictions, 0]
    # Scale and remove the last element --don't know why it's with zeroes.
    X_scaled = np.array([params['x_scaler'].fit_transform(X[i]) for i in range(X.shape[0])])[:-1]
    Y_scaled = params['y_scaler'].fit_transform(Y)[:-1]
    # Split in training and test
    return split(X_scaled, Y_scaled, params['num_testcases'])


#
