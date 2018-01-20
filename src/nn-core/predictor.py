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

(y2, yhat2, num_errors2) = lstm.predict(
    'params_8_indicators.yaml', '2yr_1hr_08inds_20e_20180120',
    dataset_name='ibex_1hr_2y.csv')
print(''.join(map(str, y2)))
plot.prediction(y2, yhat2, num_errors2)

(y3, yhat3, num_errors3) = lstm.predict(
    'params_all_indicators.yaml', '2yr_1hr_29inds_40e_20180120',
    dataset_name='ibex_1hr_2y.csv')
print(''.join(map(str, y3)))
plot.prediction(y3, yhat3, num_errors3)

y_avg = ((yhat2 + yhat3) / 2.0)
rmse, num_errors_avg = compute.error(y2, y_avg)
plot.prediction(y2, y_avg, num_errors_avg)
#
# -- Single value prediction.
#
prediction = lstm.single_predict(model1, X_test[31], Y_test[31], params)
print(prediction)
