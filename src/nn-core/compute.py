import math
from sklearn.metrics import mean_squared_error
import numpy as np


def tendency_errors(Y, Yhat):
    """
    Compute the error in tendency (sign of future value minus present value)
    when making a prediction.
    """
    num_errors = 0
    for idx in range(1, len(Y)):
        yhat_trend = np.sign(Yhat[idx]-Y[idx-1])
        y_trend = np.sign(Y[idx]-Y[idx-1])
        error = int(yhat_trend == y_trend)
        if error == 0:
            num_errors += 1
    return num_errors


def error(inv_y, inv_yhat):
    """
    Compute the RMSE between the prediction and the actual values.
    """
    rmse = math.sqrt(mean_squared_error(inv_y, inv_yhat))
    return rmse, tendency_errors(inv_y, inv_yhat)
