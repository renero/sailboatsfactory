import pandas
from pandas import read_csv
from numpy import empty, inf
from pathlib import Path
from os.path import join

from parameters import param_set


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
    params['raw_numcols'] = raw_dataset.shape[1]
    return (raw_dataset)


def normalize(df):
    """
    Normalize input according to the price change ratio

    n_i = ( p_i / p_0 ) - 1

    where n_i is the new normalized value, p_i is the i'th value of the df
    data, and p_0 is the first value.

    :param df: A pandas dataframe with multiple columns. All of them will
    be normalized.
    :returns: A new dataframe with all columns normalized.
    """
    normalized = pandas.DataFrame(pandas.np.empty(df.shape),
                                  columns=df.columns.tolist())
    for column in df.columns.tolist():
        p0 = df.loc[0, column]
        print('p0 for col({}) = {}'.format(column, p0))
        normalized.loc[:, column] = (df.loc[:, column] / p0) - 1.0
    normd_clean = normalized.loc[1:, :].reset_index().drop(['index'], axis=1)
    return normd_clean.fillna(0.0).replace([inf, -inf], 0)


def denormalize(normalized, params):
    denormalized = pandas.DataFrame(
        pandas.np.empty((params['adj_numrows'], params['adj_numcols'])),
        columns=normalized.columns)
    for column in normalized.columns.tolist():
        p0 = params['p0'][column]
        denormalized.loc[0, column] = p0
        denormalized.loc[1:, column] = p0 * (
            normalized.loc[1:, column] + 1.0)
    return denormalized


def denormalize_vector(vector, column_name, params):
    """
    Denormalize a vector of values, previosly normalized by the 'normalize'
    method. Denormaliztion needs to know what is the first value in the series
    that was used as reference, that is stored in params['p0']

    p_i = p_0 * (n_i + 1.0)

    :param vector: The series of numbers that needs to be denormalized.
    :param column_name: The name of the vector, and is used to access p0 value
    :param params: The cache with all execution parameters
    :returns: the vector, denormalized.
    """
    denormalized = pandas.DataFrame(
        pandas.np.empty((vector.shape[0], 1)),
        columns=[column_name])
    p0 = params['p0'][column_name]
    denormalized.loc[:, column_name] = p0 * (vector + 1.0)
    return denormalized


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

    # Save the first row to later re-construct normalized values.
    p0 = raw.loc[0, :]
    params['p0'] = p0
    # raw_prepared will be the data normalized used to split X and Y.
    raw_prepared = raw.copy()
    print('Raw shape:', raw_prepared.shape)
    raw_prepared = normalize(raw)
    print('Normalized shape:', raw_prepared.shape)

    # Setup the windowing of the dataset.
    num_samples = raw_prepared.shape[0]
    num_features = raw_prepared.shape[1]
    num_predictions = params['lstm_predictions']
    num_timesteps = params['lstm_timesteps']
    num_frames = num_samples - (num_timesteps + num_predictions) + 1
    # Update internal cache of parameters
    params['num_samples'] = num_samples
    params['num_features'] = num_features
    params['num_frames'] = num_frames
    # print('Num samples.....:', num_samples)
    # print('Num features....:', num_features)
    # print('Num frames......:', num_frames)
    # print('Num timesteps...:', num_timesteps)
    # print('Num predictions.:', num_predictions)
    # print('Batch size......:', params['lstm_batch_size'])

    # Build the 3D array (num_frames, num_timesteps, num_features)
    X = empty((num_frames, num_timesteps, num_features))
    Y = empty((num_frames, num_predictions))
    print('X[{}], Y[{}]'.format(X.shape, Y.shape))
    for i in range(num_samples - num_timesteps):
        X[i] = raw_prepared.loc[i:i + num_timesteps - 1, :]
        Y[i] = raw_prepared.loc[
                i + num_timesteps, 'tickcloseprice'
               ]
    return split(X, Y, params['num_testcases'])
