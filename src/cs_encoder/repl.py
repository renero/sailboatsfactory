import numpy as np
import pandas as pd

from cs_encoder.cs_encoder import CSEncoder
from cs_encoder.ticks import Ticks
from nn_cse.cs_plot import CSPlot

target_cols = [
    'tickopenprice', 'tickmaxprice', 'tickminprice', 'tickcloseprice'
]
ohlc_tags = ['o', 'h', 'l', 'c']
df = Ticks.read_ohlc('../data/100.csv', target_cols, ohlc_tags)
cse = []
for index in range(0, df.shape[0]):
    cse.append(CSEncoder(np.array(df.iloc[index])))
    cse[index].encode_body()
    cse[index].encode_movement(cse[index - 1])
CSEncoder.save(cse, '../data/100.cse')

# Plot the first two cse elements
CSPlot().plot(df[0:2], ohlc_names=ohlc_tags)

# TODO: Reconstruir un CS a partir de la codificación de las DELTAS
tick0 = [cse[0].min, cse[0].high, cse[0].low, cse[0].max]
tick1 = [cse[1].open, cse[1].high, cse[1].low, cse[1].close]
mm = cse[0].hl_interval_width
rtick1 = [
    tick0[0] + (CSEncoder.decode_movement(cse[1].encoded_delta_min) * mm),
    tick0[1] + (CSEncoder.decode_movement(cse[1].encoded_delta_high) * mm),
    tick0[2] + (CSEncoder.decode_movement(cse[1].encoded_delta_low) * mm),
    tick0[3] + (CSEncoder.decode_movement(cse[1].encoded_delta_max) * mm)
]

prd = pd.DataFrame(columns=ohlc_tags)
prd.loc[0] = tick0
prd.loc[1] = rtick1
prd.loc[2] = tick1
CSPlot().plot(prd, ohlc_names=ohlc_tags)
