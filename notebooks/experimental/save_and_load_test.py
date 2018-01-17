from numpy.random import seed
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

# load json and create model
params = parameters.read()
raw = data.read(params)
print('Original dataset num samples:', raw.shape)
adjusted = parameters.adjust(raw, params)
X_train, Y_train, X_test, Y_test = data.prepare(adjusted, params)

# Build the model and train it.
params['lstm_batch_size'] = 1
model = lstm.build(params)

# load weights into new model
model.load_weights("20180116_0438.h5")
print("Loaded weights from disk")

print('Actual:', params['y_scaler'].inverse_transform(Y_test[31]))

# Plot the test values for Y, and Y_hat, without scaling (inverted)
Y_hat = model.predict(X_test[31].reshape((1, 6, 8)), batch_size=params['lstm_batch_size'])
print('Prediction:', params['y_scaler'].inverse_transform(Y_hat))
