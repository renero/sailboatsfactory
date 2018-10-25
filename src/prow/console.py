from cs_encoder import CSEncoder
from ticks import Ticks
from params import Params
from cs_api import predict_next_close, prepare_nn, prepare_datasets, \
    predict_testset

import matplotlib.pyplot as plt
import numpy as np

params = Params()
ticks = Ticks().read_ohlc()
encoder = CSEncoder().fit(ticks)
# TODO: La lista de CSEs no tienen que ser objetos del mismo tipo que el encoder
cse = encoder.ticks2cse(ticks)
dataset = prepare_datasets(encoder, cse, params)
nn = prepare_nn(dataset, params)
preds = predict_testset(dataset, encoder, nn, params)

# Single prediction case
# errors = []
# for i in range(100):
#     start = 2000 + i
#     end = start + params._window_size
#     tick = ticks.iloc[start:end]
#     real_close = ticks.iloc[end:end + 1]['c'].values[0]
#     next_close = predict_next_close(tick, encoder, nn, params)
#     errors.append(abs(next_close - real_close))
#
# plt.plot(errors)
# med = np.median(errors)
# std = np.std(errors)
# plt.axhline(med, linestyle=':', color='red')
# plt.axhline(med + std, linestyle=':', color='green')
# plt.show()
# plt.hist(errors, color='blue', edgecolor='black', bins=int(100 / 2))
# plt.show()

#
# EOF
#
