from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


class DataPrep(object):

    X_train = None
    y_train = None
    X_test = None
    y_test = None

    def __init__(self, data, window_size, test_size):
        self.data = data.copy()
        self._window_size = window_size
        self._num_categories = data.shape[1]
        series = data.copy()
        series_s = series.copy()
        for i in range(window_size):
            series = pd.concat([series, series_s.shift(-(i + 1))], axis=1)
        series.dropna(axis=0, inplace=True)
        train, test = train_test_split(
            series, test_size=test_size, shuffle=False)
        self.X_train, self.y_train = self.reshape(np.array(train))
        self.X_test, self.y_test = self.reshape(np.array(test))

    def reshape(self, data):
        num_entries = data.shape[0] * data.shape[1]
        timesteps = self._window_size + 1
        num_samples = int((num_entries / self._num_categories) / timesteps)
        train = data.reshape((num_samples, timesteps, self._num_categories))
        X_train = train[:, 0:self._window_size, :]
        y_train = train[:, -1, :]
        return X_train, y_train
