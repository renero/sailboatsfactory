from numpy.random import seed
from numpy import log
from tensorflow import set_random_seed

import compute
import data
import lstm
from model import setup, save
import parameters
import plot


# %matplotlib inline
%load_ext autoreload
%autoreload 2

# Initialization of seeds
set_random_seed(2)
seed(2)

# Read the parameters, dataset and then adjust everything
# to produce the training and test sets with the correct
# batch size splits.
params = parameters.read('params_8i_1lyr_3yrs.yaml')
raw = data.read(params)
adjusted = parameters.adjust(raw, params)



# Find a problem with INFINITE numbers...
import numpy as np
import matplotlib.pyplot as plt
print(np.any(np.isnan(adjusted)))
print(np.all(np.isfinite(adjusted)))
np.where(np.all(np.isnan(adjusted), axis=1))[0]

X_train, Y_train, X_test, Y_test = data.prepare(adjusted, params)
