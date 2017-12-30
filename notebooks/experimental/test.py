import numpy as np
import data
from yaml import load
from os.path import join
from sklearn.preprocessing import MinMaxScaler
import data
import plot


def read_parameters():
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

params = read_parameters()
raw = data.read(params)
raw.shape
plot.features(raw)
