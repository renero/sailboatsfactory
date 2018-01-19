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

(y1, yhat1, num_errors1) = lstm.predict(
    'params_all_indicators.yaml', 'all_indicators20180118_1106',
    dataset_name='ibex_1hr_2y_rounded.csv')
plot.prediction(y1, yhat1, num_errors1)
plt.plot(y1)
plt.show()

(y2, yhat2, num_errors2) = lstm.predict(
    'params_8_indicators.yaml', '8indicators20180117_1200',
    dataset_name='ibex_1hr_2y_rounded.csv')
plot.prediction(y2, yhat2, num_errors2)
plt.plot(y2)
plt.show()

(y3, yhat3, num_errors3) = lstm.predict(
    'params_newinds_2y.yaml', 'newinds_2y_20180118_2200',
    dataset_name='ibex_1hr_2y_rounded.csv')
plot.prediction(y3, yhat3, num_errors3)
plt.plot(y3)
plt.show()

y_avg = ((yhat2 + yhat3) / 2.0)
rmse, num_errors_avg = compute.error(y1, y_avg)
plot.prediction(y1, y_avg, num_errors_avg)
#
# -- Single value prediction.
#
prediction = lstm.single_predict(model1, X_test[31], Y_test[31], params)
print(prediction)


plt.figure(figsize=(8, 6))
plt.plot(y2)
plt.plot(y3)
plt.show()
