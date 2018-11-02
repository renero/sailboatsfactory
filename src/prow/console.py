#!/usr/bin/env python
import numpy as np
import pandas as pd

from cs_encoder import CSEncoder
from ticks import Ticks
from params import Params
from cs_api import train_nn, prepare_datasets, single_prediction
from cs_utils import random_tick_group
from cs_plot import CSPlot as plot

params = Params()
ticks = Ticks().read_ohlc()


if params.do_train is True:
    encoder = CSEncoder().fit(ticks)
    cse = encoder.ticks2cse(ticks)
    dataset = prepare_datasets(encoder, cse, params.subtypes)
    nn = train_nn(dataset, params.subtypes)
    encoder.save()
else:
    tick_group = random_tick_group(ticks, params.max_tick_series_length + 1)
    plot.candlesticks(tick_group, ohlc_names=params._ohlc_tags)
    prediction = single_prediction(tick_group[:-1])
    prediction['mean'] = prediction.mean(axis=1)
    print('real[{:.4f}];avg<{:04f}>'.format(tick_group['c'][-1],
                                            prediction['mean'][0]))
#
# EOF
#
