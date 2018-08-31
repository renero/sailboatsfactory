import numpy as np
import pandas as pd

class Ticks(object):

    @staticmethod
    def read_ohlc(filepath, target_cols, ohlc_cols, normalize=True):
        cols_mapper = dict(zip(target_cols, ohlc_cols))
        df = pd.read_csv('../data/100.csv', delimiter='|')
        df = df[list(cols_mapper.keys())].rename(index=str, columns=cols_mapper)
        max_value = df.values.max()
        min_value = df.values.min()

        def normalize(x):
            return (x - min_value) / (max_value - min_value)

        df = df.applymap(np.vectorize(normalize))
        return df

    @staticmethod
    def new_ohlc(values, columns):
        df = pd.Series([values], columns=columns)
        return df
