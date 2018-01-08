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

# Read the parameters, dataset and then adjust everything
# to produce the training and test sets with the correct
# batch size splits.
params = parameters.read()
raw = data.read(params)
print('Original dataset num samples:', raw.shape)
adjusted = parameters.adjust(raw, params)
X_train, Y_train, X_test, Y_test = data.prepare(adjusted, params)

# Build the model and train it.
# model = lstm.load('20180106_0133.h5')
model = lstm.build(params)
train_loss = lstm.fit(model, X_train, Y_train, params)
plot.history(train_loss)

# Plot the test values for Y, and Y_hat, without scaling (inverted)
Y_hat = model.predict(X_test, batch_size=params['lstm_batch_size'])
rmse, num_errors = compute.error(Y_test, Y_hat)
plot.prediction(params['y_scaler'].inverse_transform(Y_test),
                params['y_scaler'].inverse_transform(Y_hat),
                num_errors)

# Save the model
# saved_model_name = lstm.save(model)
