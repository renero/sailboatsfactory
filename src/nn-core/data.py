import pandas
from pandas import read_csv
from numpy import empty
import numpy as np
from pathlib import Path
from os.path import join


def read(params, dataset_file=''):
    """
    If dataset_file is specified then whatever training_file is specified
    in the parameter dictionary is override by it.
    """
    home_path = str(Path.home())
    load_folder = join(join(home_path, params['project_path']),
                       params['data_path'])
    if dataset_file is '':
        print("Reading dataset:", params['training_file'])
        file_name = join(load_folder, params['training_file'])
    else:
        print("Reading dataset:", dataset_file)
        file_name = join(load_folder, dataset_file)
    raw_dataset = read_csv(
        file_name,
        header='infer',
        delimiter=params['delimiter'],
        usecols=params['columNames'])
    params['raw_numrows'] = raw_dataset.shape[0]
    return (raw_dataset)


def diff(a, interval=1):
    """
    Given a 2D array (a), compute the one resulting from differentiating each
    element by the one at "interval" distance from it, only for the first
    column:

    [  1.   1.]
    [  2.   2.]  =>  [ 1.   2.]
    [  4.   3.]  =>  [ 2.   3.]
    [  7.   4.]  =>  [ 3.   4.]
    [ 11.   5.]  =>  [ 4.   5.]

    """
    if a.ndim is not 2:
        raise ValueError(
            'Differentiating tensor with wrong number of dimensions ({:d})'.
            format(a.ndim))
    return np.concatenate((
                         (a[interval:, 0:1] - a[:-interval, 0:1]),
                          a[interval:, 1:]),
                          axis=1)


def inverse_diff(a_diff, a, interval=1):
    """
    Inverts the operation at 'diff', needing the original vector.
    """
    if a_diff.ndim is not 2 or a.ndim is not 2:
        raise ValueError(
            'Differentiating with wrong number of dimensions({:d})'.format(
                a.ndim))
    col0 = np.concatenate((a[0:interval, 0:1], (a_diff[:, 0:1] + a[:-1, 0:1])),
                          axis=0)
    return np.concatenate((col0, a[:, 1:]), axis=1)


def split(X, Y, num_testcases):
    """
    Splits X, Y horizontally to separate training from test, as specified
    by the number of test cases.
    """
    print('Splitting {:d} test cases'.format(num_testcases))
    X_train = X[:-num_testcases, ]
    Y_train = Y[:-num_testcases, ]
    X_test = X[-num_testcases:, ]
    Y_test = Y[-num_testcases:, ]
    print('X_train[{}], Y_train[{}]'.format(X_train.shape, Y_train.shape))
    print('X_test[{}], Y_test[{}]'.format(X_test.shape, Y_test.shape))
    return X_train, Y_train, X_test, Y_test


def prepare(raw, params):
    """
    Takes the data series as a sequence of rows, with "num_features" features
    on each line, and transform it into a 3D array, where the first dimension
    is the number of sliding windows formed (num_frames), of size
    "num_timesteps", each of them with "num_features" features.

    For num_timesteps=2, 3 features on each row, and making 1 prediction in
    the future, the original raw data (5 x 3) is transformed into:
    --> (3 x (2+1) x 3) =
        (num_frames x (num_timesteps + num_predictions) x num_features):

    [1,2,3]     ------- X ------ -- Y --
    [1,4,5]     [[1,2,3],[1,4,5],[2,_,_]]
    [2,6,3]  => [[1,4,5],[2,6,3],[4,_,_]]
    [4,2,3]     [[2,6,3],[4,2,3],[5,_,_]]
    [5,3,8]
    """
    # Diff, first of all.
    pandas.options.mode.chained_assignment = None
    raw_columns = raw.columns.tolist()
    for logable_column in params['stationarizable']:
        if logable_column in raw_columns:
            raw.loc[:, logable_column] = np.log1p(raw.loc[:, logable_column])
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
    print('X[{}], Y[{}]'.format(X.shape, Y.shape))
    for i in range(num_samples - num_timesteps):
        X[i] = non_stationary[i:i + num_timesteps, ]
        Y[i] = non_stationary[
                i + num_timesteps:i + num_timesteps + num_predictions, 0
               ]
    # Scale
    X_scaled = np.array([params['x_scaler'].fit_transform(X[i])
                        for i in range(X.shape[0])])
    Y_scaled = params['y_scaler'].fit_transform(Y)
    # Split in training and test
    return split(X_scaled, Y_scaled, params['num_testcases'])


def recover_Ytest(Y_test, raw, params):
    """
    From the test values of Y, returns the original values, after
    inverting the scaling, and the diff operations. We need the
    original raw values for that.
    """
    y_unscaled = params['y_scaler'].inverse_transform(Y_test)
    y_undiff = inverse_diff(
                y_unscaled,
                raw[-(params['num_testcases']+1):, 0:1])
    return y_undiff[-params['num_testcases']:]
