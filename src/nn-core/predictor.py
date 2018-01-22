from numpy.random import seed
from numpy import log, exp
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

(params1, _, y1, yhat1, num_errors1) = lstm.predict('params_8i_4lyrs.yaml')
plot.prediction(params1['y_scaler'].inverse_transform(exp(y1)),
                params1['y_scaler'].inverse_transform(exp(yhat1)),
                num_errors1)

(params2, model2, y2, yhat2, num_errors2) = lstm.predict('params_8i.yaml')
plot.prediction(params2['y_scaler'].inverse_transform(exp(y2)),
                params2['y_scaler'].inverse_transform(exp(yhat2)),
                num_errors2)

y_avg = ((yhat1 + yhat2) / 2.0)
rmse, num_errors_avg = compute.error(y1, y_avg)
plot.prediction(y1, y_avg, num_errors_avg)


#
# -- Single value prediction.
#
# prediction = lstm.single_predict(model1, X_test[31], Y_test[31], params)
# print(prediction)
