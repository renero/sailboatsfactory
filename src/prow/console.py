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
    tick_group = random_tick_group(ticks, params.max_tick_series_length)
    nn = load_nn(params.model_names, params.subtypes)
    for name in params.model_names:
        params.log.info(name)
        nn_encoder = CSEncoder().load(params.model_names[name]['encoder'])
        next_close = predict_close(tick_group, nn_encoder, nn[name], params)
        params.log.highlight('Close pred.{}: {:.4f}'.format(name, next_close))

#
# EOF
#
