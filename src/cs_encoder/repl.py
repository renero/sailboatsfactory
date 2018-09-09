import numpy as np
import pandas as pd

from cs_encoder.cs_encoder import CSEncoder
from cs_encoder.onehot_encoder import OnehotEncoder
from cs_encoder.ticks import Ticks
from nn_cse.cs_plot import CSPlot
from pprint import pprint

target_cols = [
    'tickopenprice', 'tickmaxprice', 'tickminprice', 'tickcloseprice'
]
ohlc_tags = ['o', 'h', 'l', 'c']
ticks = Ticks.read_ohlc('../data/100.csv', target_cols, ohlc_tags)
cse = CSEncoder.encode_ticks(ticks)
CSPlot().plot(ticks, ohlc_names=ohlc_tags)

# Save encodings to CSE file
CSEncoder.save(cse, '../data/100.cse')

# Reconstruct the tick from the encodings
rec_ticks = CSEncoder.decode_ticks(cse, ohlc_tags)
CSPlot().plot(rec_ticks, ohlc_names=ohlc_tags)

# One hot encoding.

ohe = OnehotEncoder(signed=True).fit_from_dictionary(
    np.array(CSEncoder._def_prcntg_encodings))

values_to_encode = np.array([cse[i].encoded_delta_open for i in range(0, 4)])
pprint(values_to_encode)
encoded = ohe.transform(values_to_encode)
pprint(encoded[0])

decoded = ohe.decode(encoded[0])
pprint(decoded)
