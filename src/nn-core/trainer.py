from numpy.random import seed
from numpy import repeat
from tensorflow import set_random_seed

import compute
import data
import lstm
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
params = parameters.read('params_8_indicators.yaml')
adjusted = parameters.adjust(data.read(params), params)
X_train, Y_train, X_test, Y_test = data.prepare(adjusted, params)
# model = lstm.load('20180117_1200', params)
model = lstm.build(params)
train_loss = lstm.fit(model, X_train, Y_train, params)
(model_name, net_name) = lstm.save(model)

# Build the predictor model, with batch_size = 1, and save it.
bs = params['lstm_batch_size']
params['lstm_batch_size'] = 1
pred_model = lstm.build(params)
lstm.save(pred_model, prefix='pred_', save_weights=False)
params['lstm_batch_size'] = bs
del pred_model
