#!/usr/bin/env python

from cs_encoder import CSEncoder
from ticks import Ticks
from params import Params
from cs_api import load_encoders, train_nn, prepare_datasets, single_prediction, \
    load_nn, add_supervised_info
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
    nn = load_nn(params.model_names, params.subtypes)
    encoder = load_encoders(params.model_names)

    tick_group = random_tick_group(ticks, params.max_tick_series_length + 1)
    prediction = single_prediction(tick_group[:-1], nn, encoder, params)
    prediction = add_supervised_info(prediction, tick_group['c'][-1], params)

    print('real[{:.4f}];avg<{:04f}>;avg.diff:{:.4f};wnr:{}'.format(
        tick_group['c'][-1],
        prediction['mean'][0],
        prediction['avg_diff'][0],
        prediction['winner'][0]
    ))
#
# EOF
#
