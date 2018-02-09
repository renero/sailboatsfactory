import numpy
from numpy.random import seed
from numpy import log, exp, sign
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

(params4, _, yraw, y4, yhat4, num_errors4) =\
    lstm.predict('params_3y_1L256_09i.yaml')
plot.prediction(y4, yhat4, yraw, num_errors4, params4,
                inv_scale=False, inv_diff=False, inv_log=False)
# Check individual predictions
#
# Estoy intentando producir como salida de cada predicción un valor para el
# precio. Para ello cojo la predicción individual y la intento invertir
# aplicando los mismos pasos que para la grafica de arriba, pero lo que obtengo
# es un valor absurdamente pequeño...
# Quiza debería probar a hacer predicciones individuales,
# O =>>> comparar cada predicción de tendencia con la que de verdad se produce
# arriba en el grafico, o con los valores invertidos, que sería lo mismo.
#
# ...do it, please.
for idx in range(len(yhat4)):
    x = idx
    y = yhat4[idx]
    if idx > 0:
        yhat_trend = sign(yhat4[idx]-y4[idx-1])
        y_trend = sign(y4[idx]-y4[idx-1])

        y_original = params4['y_scaler'].inverse_transform(
            y4[idx].reshape(-1, 1))
        y_original = data.inverse_diff(
            y_original, numpy.array([y4[idx-1], y4[idx]]))
        y_original = numpy.expm1(y_original)

        if numpy.array_equal(yhat_trend, y_trend) is False:
            err_msg = "ERR"
        else:
            err_msg = "OK!"
        print('{:.4f} -> {:.4f} / {:.4f} => [{:+} / {:+}] {} ··> {:.4f}'
              .format(float(y4[idx-1]), float(y4[idx]), float(yhat4[idx]),
                      int(y_trend), int(yhat_trend), err_msg,
                      float(y_original[-1])))
    else:
        print('{:.4f}'.format(float(y4[idx])))

y_avg = ((yhat1 + yhat2 + yhat3) / 3.0)
rmse, num_errors_avg = compute.error(y1, y_avg)
plot.prediction(y1, y_avg, num_errors_avg, params1)


#
# -- Single value prediction.
#
# prediction = lstm.single_predict(model1, X_test[31], Y_test[31], params)
# print(prediction)















































#,jslkh
