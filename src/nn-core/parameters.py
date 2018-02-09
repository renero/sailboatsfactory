from os.path import join
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from yaml import load


def read(filename='params.yaml'):
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
    return my_params


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


def find_largest_divisor(x, divisor, params, all=False):
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
    new_testshape = find_largest_divisor(params['num_testcases'],
                                         params['lstm_batch_size'],
                                         params,
                                         all=True)
    params['num_testcases'] = new_testshape
    new_shape = find_largest_divisor(raw.shape[0],
                                     params['lstm_batch_size'],
                                     params,
                                     all=False)
    print('Reshaping raw from [{}] to [{}]'.
          format(raw.shape, raw[-new_shape:].shape))
    return raw[-new_shape:]


def param_set(params, param):
    """
    Check if 'param' is set in the parameters, and if so, returns if is set
    to True. Otherwise returns False.
    """
    if param in params:
        if params[param] is True:
            return True
    return False
