import numpy as np
from cs_encoder.candlestick import Candlestick
from cs_encoder.ohlc_reader import read_ohlc
from nn_cse.cs_plt import Csplt

target_cols = [
    'tickopenprice', 'tickmaxprice', 'tickminprice', 'tickcloseprice'
]
ohlc_tags = ['o', 'h', 'l', 'c']
df = read_ohlc('../data/100.csv', target_cols, ohlc_tags, normalize=True)

cse = [Candlestick(np.array(df.iloc[i])) for i in range(df.shape[0])]
for cs in cse:
    cs.encode_body()
for index in range(1, len(cse)):
    cse[index].encode_movement(cse[index-1])

Csplt().plot(df[0:2], ohlc_names=ohlc_tags)

# TODO: Reconstruir un CS a partir de la codificación de las DELTAS
