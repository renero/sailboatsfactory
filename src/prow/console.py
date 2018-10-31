#!/usr/bin/env python

from cs_encoder import CSEncoder
from ticks import Ticks
from params import Params
from cs_api import predict_close, train_nn, load_nn, prepare_datasets
from cs_utils import random_tick_group

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
    for name in params.model_names:
        params.log.info(name)
        encoder = CSEncoder().load(params.model_names[name]['encoder'])

    all_agree = 0
    for i in range(50):
        preds = []
        tick_group = random_tick_group(ticks, params.max_tick_series_length)
        for name in params.model_names:
            next_close = predict_close(tick_group, encoder, nn[name], params)
            preds.append(next_close)
            params.log.highlight('Close pred.{}: {:.4f}'.format(name, next_close))
        if preds[0] == preds[1] and preds[1] == preds[2]:
            all_agree += 1
            print('+', end='')
        else:
            print('-', end='')

#
# EOF
#
