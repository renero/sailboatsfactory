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
# One hot encoding.
#
ohe = OnehotEncoder(signed=True).fit_from_dictionary(
    encoder.movement_dictionary())
pprint(cse_shifts.values[0:3])
encoded = ohe.transform(cse_shifts.values)
oh_shifts = encoded.reshape((encoded.shape[0] * encoded.shape[1]),
                            encoded.shape[2])
oh_shifts = oh_shifts.transpose()
oh_shifts.shape
oh_shifts[0]
pprint(encoded[0])

#
# Encode/Decode body part.
#
ohe = OnehotEncoder(signed=True).fit_from_dictionary(encoder.body_dictionary())
values_to_encode = np.array([cse[i].encoded_body for i in range(n)])
pprint(values_to_encode)
encoded = ohe.transform(values_to_encode)
pprint(encoded[0])
decoded = ohe.decode(encoded[0])
pprint(decoded)

#
# Reverse Encoding to produce ticks from CSE
#
cse_codes = encoder.read_cse(cse_file, cse_tags)
rec_ticks = encoder.cse2ticks(cse_codes.iloc[:n, ], ohlc_tags)
# CSPlot().plot(rec_ticks.iloc[:n, ], ohlc_names=ohlc_tags)
