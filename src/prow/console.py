#!/Users/renero/anaconda3/envs/py36/bin/python

import os
import sys

import tensorflow as tf
import pandas as pd
import numpy as np

from cs_api import prepare_datasets, train_nn, load_nn, load_encoders, \
    single_prediction, add_supervised_info
from cs_encoder import CSEncoder
from cs_logger import CSLogger
from cs_utils import random_tick_group, valid_output_name
from params import Params
from ticks import Ticks

tf.compat.v1.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

np.random.seed(1)
params = Params(args=sys.argv)
log = CSLogger(params._log_level)
ticks = Ticks()
ohlc_data = ticks.read_ohlc()

if params.do_train is True:
    encoder = CSEncoder().fit(ohlc_data)
    cse = encoder.ticks2cse(ohlc_data)
    dataset = prepare_datasets(encoder, cse, params.subtypes)
    nn = train_nn(dataset, params.subtypes)
    encoder.save()
else:
    nn = load_nn(params.model_names, params.subtypes)
    encoder = load_encoders(params.model_names)
    predictions = pd.DataFrame([])

    if params._predict_training:
        # for from_idx in range(0, ticks.shape[0] - params._window_size + 1):
        for from_idx in range(0, 50 - params._window_size + 1):
            tick_group = ohlc_data.iloc[from_idx:from_idx + params._window_size]
            prediction = single_prediction(tick_group, nn, encoder, params)
            prediction = add_supervised_info(
                prediction,
                ohlc_data.iloc[from_idx + params._window_size]['c'],
                params)
            predictions = predictions.append(prediction)
        predictions = ticks.scale_back(predictions)
    else:
        for i in range(10):
            tick_group = random_tick_group(ticks,
                                           params.max_tick_series_length + 1)
            prediction = single_prediction(tick_group[:-1], nn, encoder, params)
            prediction = add_supervised_info(prediction, tick_group['c'][-1],
                                             params)
            predictions = predictions.append(prediction)

    if params._save_predictions is True:
        # Reorder columns to set 'actual' in first position
        actual_position = list(predictions.columns).index('actual')
        avg_position = list(predictions.columns).index('avg')
        columns = [actual_position] + [i for i in
                                       range(actual_position)] + [avg_position]
        predictions = predictions.iloc[:, columns]

        # Find a valid filename and save everything
        filename = valid_output_name(
            filename='predictions_1y5y_w10',
            path=params._predictions_path,
            extension='csv')
        predictions.to_csv(filename, index=False)
        log.info('predictions saved to: {}'.format(filename))

#
# EOF
#
