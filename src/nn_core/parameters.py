from os.path import join
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from yaml import load


import data


def initialize(filename='params.yaml'):
    """
    Reads a YAML file within the CWD of the current notebook to read all the
    params from there.
    """
    home_path = str(Path.home())
    project_path = 'Documents/SideProjects/sailboatsfactory'
    work_path = 'src/nn-core'
    params_path = join(home_path, join(project_path, work_path))
    yaml_file = join(params_path, filename)
    print("Reading parameters from:", filename)
    with open(yaml_file, 'r') as f:
        my_params = load(f)
    my_params['x_scaler'] = MinMaxScaler(feature_range=(-1, 1))
    my_params['y_scaler'] = MinMaxScaler(feature_range=(-1, 1))

    raw = data.read(my_params)
    adjusted = adjust(raw, my_params)

    return adjusted, my_params


def valid_samples(x, params, all=False):
    """
    Given a candidate number for the total number of samples to be considered
    for training and test, this function simply substract the number of
    test cases, predictions and timesteps from it.
    """
    if all is True:
        return x
    return (x - params['num_testcases']
            - params['lstm_timesteps']
            - params['lstm_predictions'])


def find_largest_divisor(x, params, all=False):
    """
    Compute a number lower or equal to 'x' that is divisible by the divsor
    passed as second argument. The flag 'all' informs the function whether
    the number of samples can be used as such (all=True) or it must be
    adjusted substracting the number of test cases, predictions and
    num_timesteps from it.
    """
    found = False
    while x > 0 and found is False:
        if valid_samples(x, params, all) % params['lstm_batch_size'] is 0:
            found = True
        else:
            x -= 1
    return x


def adjust(raw, params):
    """
    Given a raw sequence of samples, it determines the correct number of
    samples that can be used, given the amount of test cases requested,
    the timesteps, the nr of predictions, and the batch_size.
    Returns the raw sequence of samples adjusted, by removing the first
    elements from the array until shape fulfills TensorFlow conditions.
    """
    new_testshape = find_largest_divisor(
        params['num_testcases'], params, all=True)
    print('Reshaping TEST from [{}] to [{}]'.
          format(params['num_testcases'], new_testshape))
    params['num_testcases'] = new_testshape
    new_shape = find_largest_divisor(
        raw.shape[0], params, all=False)
    print('Reshaping RAW from [{}] to [{}]'.
          format(raw.shape, raw[-new_shape:].shape))
    new_df = raw[-new_shape:].reset_index().drop(['index'], axis=1)
    params['adj_numrows'] = new_df.shape[0]
    params['adj_numcols'] = new_df.shape[1]

    # Setup the windowing of the dataset.
    params['num_samples'] = raw.shape[0]
    params['num_features'] = raw.shape[1]
    params['num_frames'] = params['num_samples'] - (
        params['lstm_timesteps'] + params['lstm_predictions']) + 1

    return new_df


def param_set(params, param):
    """
    Check if 'param' is set in the parameters, and if so, returns if is set
    to True. Otherwise returns False.
    """
    if param in params:
        if params[param] is True:
            return True
    return False


def summary(params):
    width = 65
    print('_'*width)
    print('Data')
    print('='*width)
    str1 = '({:d}, {:d})'.format(params['raw_numrows'], params['raw_numcols'])
    str2 = '({:d}, {:d})'.format(params['nrm_numrows'], params['nrm_numcols'])
    print('{:.<10}: {:<25}{:.<10}: {}'.format('Raw', str1, 'Normalized', str2))

    str1 = '({:d}, {:d}, {:d})'.format(
        params['num_frames'], params['lstm_timesteps'], params['num_features'])
    str2 = '({:d}, {:d})'.format(
        params['num_frames'], params['lstm_predictions'])
    print('{:.<10}: {:<25}{:.<10}: {}'.format('X', str1, 'Y', str2))
    print('{:.<10}: {:<25}{:.<10}: {}'.format(
          'Samples', str(params['num_samples']),
          'Timesteps', str(params['lstm_timesteps'])))

# RAW...: (11553, 5)                  Normalized: (11552, 5)
# X.....: (11546, 6, 5)               Y.........: (11546, 1)
# Xtrain: (11446, 6, 5)               Ytrain....: (11446, 1)
# Xtest.: (100, 6, 5)                 Ytest.....: (100, 1)
