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
params = parameters.read('params_3y_1L256_09i.yaml')
raw = data.read(params)
adjusted = parameters.adjust(raw, params)

X_train, Y_train, X_test, Y_test = data.prepare(adjusted, params)
model = setup(params)
train_loss = lstm.fit(model, X_train, Y_train, params)
save(model, params, prefix='3y', additional_epocs=20)
