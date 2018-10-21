from oh_encoder import OHEncoder
from cs_encoder import CSEncoder
from ticks import Ticks
from cs_nn import Csnn
from dataset import Dataset
from params import Params
from predict import Predict
from cs_api import predict_next_close

import matplotlib.pyplot as plt
from cs_plot import CSPlot
import numpy as np
import pandas as pd

params = Params()
ticks = Ticks().read_ohlc()
cs_encoder = CSEncoder().fit(ticks, params._ohlc_tags)
cse = cs_encoder.ticks2cse(ticks)

oh_encoder = {}
cse_data = {}
oh_data = {}
dataset = {}
nn = {}
prediction = {}

for name in params._names:
    call_select = getattr(cs_encoder, 'select_{}'.format(name))
    cse_data[name] = Dataset().adjust(call_select(cse))

    call_dict = getattr(cs_encoder, '{}_dict'.format(name))
    oh_encoder[name] = OHEncoder().fit(call_dict())

    oh_data[name] = oh_encoder[name].encode(cse_data[name])
    dataset[name] = Dataset().train_test_split(oh_data[name])

    nn[name] = Csnn(dataset[name], name)

    if params._train is True:
        nn[name].build_model().train().save()
    else:
        nn[name].load(params._model_filename[name], summary=False)

    if params._predict is True:
        prediction[name] = Predict(dataset[name].X_test, dataset[name].y_test,
                                   oh_encoder[name])
        call_predict = getattr(prediction[name], '{}_batch'.format(name))
        call_predict(nn[name])

# --

# Single prediction case
errors = []
for i in range(100):
    start = 2000 + i
    end = start + params._window_size
    tick = ticks.iloc[start:end]
    real_close = ticks.iloc[end:end + 1]['c'].values[0]
    next_close = predict_next_close(tick, cs_encoder, oh_encoder, nn, params)
    errors.append(abs(next_close - real_close))

plt.plot(errors)
med = np.median(errors)
std = np.std(errors)
plt.axhline(med, linestyle=':', color='red')
plt.axhline(med+std, linestyle=':', color='green')
plt.show()
plt.hist(errors, color='blue', edgecolor='black', bins=int(100/2))
plt.show()

#
# EOF
#
