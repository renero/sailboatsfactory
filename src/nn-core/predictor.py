from numpy.random import seed
from numpy import log, exp, expm1
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

(params1, _, _, y1, yhat1, num_errors1) =\
    lstm.predict('params_2y_1L128_09i.yaml')
plot.prediction(y1, yhat1, num_errors1, params2, inv_scale=True, inv_log=True)

(params2, _, _, y2, yhat2, num_errors2) =\
    lstm.predict('params_2y_4L128_09i.yaml')
plot.prediction(y2, yhat2, num_errors2, params2, inv_scale=True, inv_log=True)

(params3, _, _, y3, yhat3, num_errors3) =\
    lstm.predict('params_3y_4L128_09i.yaml')
plot.prediction(y3, yhat3, num_errors3, params3, inv_scale=True, inv_log=True)

(params4, _, yraw, y4, yhat4, num_errors4) =\
    lstm.predict('params_3y_1L256_09i.yaml')
plot.prediction(y4, yhat4, num_errors4, params4, inv_scale=True, inv_log=True)
plot.original(yraw, yhat4, params)

y_avg = ((yhat1 + yhat2 + yhat3) / 3.0)
rmse, num_errors_avg = compute.error(y1, y_avg)
plot.prediction(y1, y_avg, num_errors_avg, params1)


#
# -- Single value prediction.
#
# prediction = lstm.single_predict(model1, X_test[31], Y_test[31], params)
# print(prediction)















































#,jslkh
