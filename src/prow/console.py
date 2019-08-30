#!/usr/bin/env pythonw

import os
import sys

import tensorflow as tf
import pandas as pd
import numpy as np

from cs_api import prepare_datasets, train_nn, load_nn, load_encoders, \
    predict_dataset, single_prediction, add_supervised_info
from cs_encoder import CSEncoder
from cs_utils import random_tick_group
from params import Params
from ticks import Ticks

tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

np.random.seed(1)
params = Params(args=sys.argv)
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

    if params._predict_training:
        model = list(params._model_names.keys())[0]
        cse = encoder[model].ticks2cse(ticks)
        dataset = prepare_datasets(encoder[model], cse, params.subtypes)
        predictions = predict_dataset(dataset,
                                      encoder[model],
                                      nn[model],
                                      split='train')
        # Time to interpret predictions!! X-/
        pass
    else:
        for i in range(10):
            tick_group = random_tick_group(ticks,
                                           params.max_tick_series_length + 1)
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
