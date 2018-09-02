import numpy as np
import pandas as pd

from cs_encoder.cs_encoder import CSEncoder

target_cols = [
    'tickopenprice', 'tickmaxprice', 'tickminprice', 'tickcloseprice'
]
ohlc_tags = ['o', 'h', 'l', 'c']

tdf = pd.DataFrame(
    data={
        'o': [0.10, 0.30, 0.30],
        'h': [0.21, 0.41, 0.41],
        'l': [0.09, 0.29, 0.29],
        'c': [0.20, 0.40, 0.40]
    })
tcs = []
for index in range(0, tdf.shape[0]):
    tcs.append(CSEncoder(np.array(tdf.iloc[index])))
    tcs[index].encode_body()
    tcs[index].encode_movement(tcs[index - 1])
