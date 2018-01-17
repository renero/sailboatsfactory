from numpy.random import seed
from numpy import repeat
from tensorflow import set_random_seed

import compute
import data
import lstm
import parameters
import plot


%matplotlib inline
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
print('Original dataset num samples:', raw.shape)
adjusted = parameters.adjust(raw, params)
X_train, Y_train, X_test, Y_test = data.prepare(adjusted, params)

# Build the model and train it.
params['lstm_batch_size'] = 1
model = lstm.build(params)

# load weights into new model
print("Loading weights from disk...")
model.load_weights("../../data/networks/20180117_0830.h5")

# Takes the input vector, to make a prediction.
input_shape = (1, params['lstm_timesteps'], len(params['columNames']))
input_vector = X_test[31].reshape(input_shape)

# Take what is the actual response.
output_value = Y_test[31]
print('Actual value:', params['y_scaler'].inverse_transform(output_value))

# Make a prediction
for i in range(0, 5):
    y_hat = model.predict(input_vector, batch_size=params['lstm_batch_size'])
    print('Prediction:', params['y_scaler'].inverse_transform(y_hat))
