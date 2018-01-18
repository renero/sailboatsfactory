from numpy.random import seed
from numpy import zeros
from tensorflow import set_random_seed
import matplotlib.pyplot as plt

import compute
import data
import lstm
import parameters
import plot


%load_ext autoreload
%autoreload 2

# Initialization of seeds
set_random_seed(2)
seed(2)

# Read the parameters, dataset and then adjust everything
# to produce the training and test sets with the correct
# batch size splits.
params = parameters.read()
raw = data.read(params)
adjusted = parameters.adjust(raw, params)
X_train, Y_train, X_test, Y_test = data.prepare(adjusted, params)
# Build the model and train it.
model = lstm.load('20180117_1200', params, prefix='pred', load_weights=True)

# Takes the input vector, to make a prediction.
input_shape = (1, params['lstm_timesteps'], len(params['columNames']))
input_vector = X_test[31].reshape(input_shape)
# Take what is the actual response.
output_value = Y_test[31]
print('Actual value:', params['y_scaler'].inverse_transform(output_value))
# Make a prediction by repeating the process 'n' times (n ~ timesteps.)
for k in range(0, 5):
    y_hat = model.predict(input_vector, batch_size=params['lstm_batch_size'])
    print('Prediction:', params['y_scaler'].inverse_transform(y_hat))

preds = zeros(32)
for i in range(0, 32):
    input_vector = X_test[i].reshape(input_shape)
    # Take what is the actual response.
    output_value = Y_test[i]
    # print('Actual value:', params['y_scaler'].inverse_transform(output_value))
    # Make a prediction
    for k in range(0, 10):
        y_hat = model.predict(input_vector, batch_size=params['lstm_batch_size'])
        # print('Prediction:', params['y_scaler'].inverse_transform(y_hat))
    preds[i] = y_hat
rmse, num_errors = compute.error(Y_test, preds)
plot.prediction(Y_test[0:32], preds, num_errors)
