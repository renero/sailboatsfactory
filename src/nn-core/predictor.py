import numpy
from numpy.random import seed
from numpy import log, exp, sign
from tensorflow import set_random_seed
import matplotlib.pyplot as plt
from os.path import join, getctime
from os import listdir
from pathlib import Path
import os

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

(params, _, yraw, y, yhat, num_errors) =\
    lstm.predict('params_3y_1L256_09i.yaml')
plot.prediction(y, yhat,
                yraw, num_errors, params,
                inv_scale=False, inv_diff=False, inv_log=False)

#
# --  Averaging predictions.
#
# y_avg = ((yhat1 + yhat2 + yhat3) / 3.0)
# rmse, num_errors_avg = compute.error(y1, y_avg)
# plot.prediction(y1, y_avg, num_errors_avg, params1)


#
# -- Single value prediction.
#
# prediction = lstm.single_predict(model1, X_test[31], Y_test[31], params)
# print(prediction)











































#,jslkh
