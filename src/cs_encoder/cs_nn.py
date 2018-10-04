import pandas as pd
import numpy as np
import yaml
import matplotlib.pyplot as plt

from datetime import datetime
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential, model_from_json
from keras.utils import to_categorical
from os.path import join, basename, splitext
from pathlib import Path
from sklearn.model_selection import train_test_split

from cs_encoder.params import Params


class ValidationException(Exception):
    pass


class Csnn(Params):

    _num_categories = 0
    _window_size = 3
    _num_predictions = 1
    _test_size = 0.1
    _dropout = 0.1
    _history = None
    _enc_data = None
    _raw_data = None
    _input_file = ''

    # metadata
    _metadata = {'period': 'unk', 'epochs': 'unk', 'accuracy': 'unk'}

    # Files output
    _output_dir = ''

    # Model design
    _l1units = 256
    _l2units = 256
    _activation = 'sigmoid'

    # Training
    _epochs = 100
    _batch_size = 10
    _validation_split = 0.1
    _verbose = 1

    # Compilation
    _loss = 'mean_squared_error'
    _optimizer = 'adam'
    _metrics = ['accuracy']

    # Results
    _history = None

    X_train = None
    y_train = None
    X_test = None
    y_test = None

    def __init__(self, dataset):
        """
        Init the class with the number of categories used to encode candles
        """
        self._metadata['dataset'] = splitext(basename(self._input_file))[0]
        self._metadata['epochs'] = self._epochs
        self.X_train = dataset.X_train
        self.X_test = dataset.X_test
        self.y_train = dataset.y_train
        self.y_test = dataset.y_test

    def build_model(self, summary=True):
        """
        Builds the model according to the parameters specified for
        dropout, num of categories in the output, window size,
        """
        model = Sequential()
        model.add(
            LSTM(
                input_shape=(self._window_size, self._num_categories),
                return_sequences=True,
                units=self._l1units))
        model.add(Dropout(self._dropout))
        model.add(LSTM(self._l2units))
        model.add(Dropout(self._dropout))
        model.add(Dense(self._num_categories, activation=self._activation))
        # model.add(Activation("tanh"))
        model.compile(
            loss=self._loss, optimizer=self._optimizer, metrics=self._metrics)
        if summary is True:
            model.summary()
        return model

    def train(self, model):
        """
        Train the model and put the history in an internal stateself.
        Metadata is updated with the accuracy
        """
        self._history = model.fit(
            self.X_train,
            self.y_train,
            epochs=self._epochs,
            batch_size=self._batch_size,
            verbose=self._verbose,
            validation_split=self._validation_split)
        self._meta['accuracy'] = self._history.history['acc']
        return self._history

    def predict(self):
        """
        Make a prediction over the internal X_test set.
        """
        self._yhat = self._model.predict(self.X_test)
        return self._yhat

    def valid_output_name(self):
        """
        Builds a valid name with the metadata and the date.
        Returns The filename if the name is valid and file does not exists,
                None otherwise.
        """
        self._filename = 'model_{}_{}_{}_{}'.format(
            datetime.now().strftime('%Y%m%d_%H%M'), self._metadata['dataset'],
            self._metadata['epochs'], self._metadata['accuracy'])
        base_filepath = join(self._output_dir, self._filename)
        output_filepath = base_filepath
        idx = 1
        while Path(output_filepath).is_file() is True:
            output_filepath = '{}_{:d}'.format(base_filepath + idx)
            idx += 1
        return output_filepath

    def load_model(self, modelname, summary=True):
        """ load json and create model """
        json_file = open('{}.json'.format(modelname), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights('{}.h5'.format(modelname))
        print("Loaded model from disk")
        loaded_model.compile(
            loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
        if summary is True:
            loaded_model.summary()
        self._model = loaded_model
        return loaded_model

    def save_model(self, modelname=None):
        """ serialize model to JSON """
        if self._metadata['accuracy'] == 'unk':
            raise ValidationException('Trying to save without training.')
        if modelname is None:
            modelname = self.valid_output_name()
        model_json = self._model.to_json()
        with open('{}.json'.format(modelname), "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self._model.save_weights('{}.h5'.format(modelname))
        print("Saved model and weights to disk")

    def plot_history(self):
        if self._history is None:
            raise ValidationException('Trying to plot without training')
        """ summarize history for accuracy and loss """
        plt.plot(self._history.history['acc'])
        plt.plot(self._history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

        plt.plot(self._history.history['loss'])
        plt.plot(self._history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
