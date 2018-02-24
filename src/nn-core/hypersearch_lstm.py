import csv

from numpy.random import seed
from tensorflow import set_random_seed

import lstm
import parameters
from data import normalize, prepare, read
from model import setup

# Initialization of seeds
set_random_seed(123)
seed(123)


hyperparams = {
    'lstm_num_epochs': [1, 2, 4, 6, 8, 16, 32, 64],
    'lstm_batch_size': [1, 2, 4, 8, 16, 32, 64],
    'lstm_timesteps': [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24],
    'lstm_layer1': [64, 128, 256, 512],
    'lstm_dropout1': [0.05, 0.075, 0.1, 0.125],
    'lstm_stateful': [True, False],
    'lstm_shuffle': [True, False],
    'lstm_forget_bias': [True, False]}

gridfile = open('grid.csv', 'w')
grid = csv.writer(gridfile, delimiter='|')
grid.writerow([
    'lstm_num_epochs', 'lstm_batch_size', 'lstm_timesteps', 'lstm_layer1',
    'lstm_dropout1', 'lstm_stateful', 'lstm_shuffle', 'lstm_forget_bias',
    'rmse', 'num_errors'])

# Set default parameters and read RAW data only once.
params = parameters.read()
for hyperparam in hyperparams.keys():
    print(hyperparam)
    params[hyperparam] = hyperparams[hyperparam][0]
raw = read(params)

# Loop through all possible combinations of hyperparams
for hyperparam in hyperparams.keys():
    print(hyperparam)
    for value in hyperparams[hyperparam]:
        print('>>> Testing ', hyperparam, ': ', value, sep='')
        params[hyperparam] = value
        #
        # s e t u p
        #
        adjusted = parameters.adjust(raw, params)
        X, Y, Xtest, ytest = prepare(normalize(adjusted, params), params)
        #
        # t r a i n i n g
        #
        model = setup(params)
        parameters.summary(params)
        model.summary()
        lstm.stateless_fit(model, X, Y, Xtest, ytest, params)
        #
        # r e b u i l d   &   p r e d i c t
        #
        pred = lstm.build(params, batch_size=1)
        pred.set_weights(model.get_weights())
        (yhat, rmse, num_errors) = lstm.range_predict(pred, Xtest, ytest, params)
        #
        # w r i t e   r e s u l t s
        #
        grid.writerow([
            params['lstm_num_epochs'], params['lstm_batch_size'],
            params['lstm_timesteps'], params['lstm_layer1'],
            params['lstm_dropout1'], params['lstm_stateful'],
            params['lstm_shuffle'], params['lstm_forget_bias'],
            rmse, num_errors])
        print(
            params['lstm_num_epochs'], ';', params['lstm_batch_size'], ';',
            params['lstm_timesteps'], ';', params['lstm_layer1'], ';',
            params['lstm_dropout1'], ';', params['lstm_stateful'], ';',
            params['lstm_shuffle'], ';', params['lstm_forget_bias'], ';',
            rmse, ';', num_errors)

gridfile.close()
