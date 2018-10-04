from cs_encoder.onehot_encoder import OnehotEncoder
from cs_encoder.cs_encoder import CSEncoder
from cs_encoder.ticks import Ticks
from cs_encoder.cs_plot import CSPlot
from nn_cse.cs_nn import Csnn

import pandas as pd
import numpy as np
from pprint import pprint

ticks_file = '../data/100.csv'
cse_file = '../data/100.cse'
target_cols = [
    'tickopenprice', 'tickmaxprice', 'tickminprice', 'tickcloseprice'
]
ohlc_tags = ['o', 'h', 'l', 'c']
cse_tags = ['b', 'o', 'h', 'l', 'c']
n = 100
LOG_LEVEL = 0

#
# Read raw data, and encode it.
#
ticks = Ticks.read_ohlc(ticks_file, target_cols, ohlc_tags)
encoder = CSEncoder(log_level=LOG_LEVEL)
encoder.fit(ticks, ohlc_tags)
cse = encoder.ticks2cse(ticks.iloc[:n, ])
encoder.save_cse(cse, cse_file)
# -> CSPlot().plot(ticks.iloc[:n, ], ohlc_names=ohlc_tags)

#
# Adjust dataset to fit into NN parameters
#
cse_nn = Csnn().init('./nn_cse/params.yaml')
cse_bodies = cse_nn.adjust(encoder.select_body(cse))
cse_shifts = cse_nn.adjust(encoder.select_movement(cse))

#
# One hot encoding
#
body_encoder = OnehotEncoder().fit_from_dictionary(encoder.body_dict())
shift_encoder = OnehotEncoder().fit_from_dictionary(encoder.move_dict())
oh_bodies = body_encoder.transform(cse_bodies.values).reshape(
    len(cse_bodies), -1)
oh_shifts = shift_encoder.transform(cse_shifts.values).reshape(
    len(cse_shifts), -1)

#
# Reverse Encoding to produce ticks from CSE
#
cse_codes = encoder.read_cse(cse_file, cse_tags)
rec_ticks = encoder.cse2ticks(cse_codes.iloc[:n, ], ohlc_tags)
# CSPlot().plot(rec_ticks.iloc[:n, ], ohlc_names=ohlc_tags)
