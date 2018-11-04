#!/usr/bin/env pythonw

from cs_api import load_encoders, train_nn, prepare_datasets, \
    single_prediction, load_nn, add_supervised_info
from cs_encoder import CSEncoder
from cs_utils import random_tick_group
from params import Params
from ticks import Ticks
import pandas as pd

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
    predictions = pd.DataFrame([])
    for i in range(10):
        tick_group = random_tick_group(ticks, params.max_tick_series_length + 1)
        prediction = single_prediction(tick_group[:-1], nn, encoder, params)
        prediction = add_supervised_info(prediction, tick_group['c'][-1],
                                         params)
        predictions = predictions.append(prediction)

    if params._save_predictions is True:
        predictions.to_csv(params._predictions_path, index=False)
    print(predictions)

#
# EOF
#
