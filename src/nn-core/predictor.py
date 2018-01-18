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

(y, yhat1, num_errors1) = lstm.predict('params_all_indicators.yaml', 'all_indicators20180118_1106')
plot.prediction(y, yhat1, num_errors1)

(y, yhat2, num_errors2) = lstm.predict('params_8_indicators.yaml', '8indicators20180117_1200')
plot.prediction(y, yhat2, num_errors2)

y_avg = ((yhat1 + yhat2) / 2.0)
rmse, num_errors3 = compute.error(y, y_avg)
plot.prediction(y, y_avg, num_errors3)
#
# -- Single value prediction.
#
prediction = lstm.single_predict(model1, X_test[31], Y_test[31], params)
print(prediction)
