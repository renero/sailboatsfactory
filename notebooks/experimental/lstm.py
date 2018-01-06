from datetime import datetime
from keras.models import load_model
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from numpy.random import seed
from tensorflow import set_random_seed


# Initialization of seeds
set_random_seed(2)
seed(2)


def build(params):
    """
    Build the LSTM according to the parameters passed. The general
    architecture is set in the code.
    """
    model = Sequential()
    # Check if my design has more than 1 layer.
    ret_seq_flag = False
    if params['lstm_numlayers'] > 1:
        ret_seq_flag = True
    print('1st layer return sequence: {:s}'.format(str(ret_seq_flag)))
    # Add input layer.
    print('Adding layer #{:d} [{:d}]'
          .format(1, params['lstm_layer{:d}'.format(1)]))
    model.add(LSTM(
            params['lstm_layer1'],
            stateful=params['lstm_stateful'],
            unit_forget_bias=params['lstm_forget_bias'],
            unroll=params['lstm_unroll'],
            batch_input_shape=(params['lstm_batch_size'],
                               params['lstm_timesteps'],
                               params['num_features']),
            return_sequences=ret_seq_flag))
    model.add(Dropout(params['lstm_dropout1']))
    # Add additional hidden layers.
    for layer in range(1, params['lstm_numlayers']):
        if (layer+1) is params['lstm_numlayers']:
            ret_seq_flag = False
        print('Adding layer #{:d} [{:d}]'.format(
            layer+1, params['lstm_layer{:d}'.format(layer+1)]))
        model.add(LSTM(params['lstm_layer{:d}'.format(layer+1)],
                       return_sequences=ret_seq_flag))
        model.add(Dropout(params['lstm_dropout{:d}'.format(layer+1)]))

    # Output layer.
    model.add(Dense(params['lstm_predictions']))
    model.compile(loss=params['lstm_loss'], optimizer=params['lstm_optimizer'])
    return model


def fit(model, X_train, Y_train, params):
    """
    Train the model passed as 1st argument, and return the train_loss
    X and Y Training values are passed.
    Parameters dictionary is also necessary.
    """
    train_loss = model.fit(
                         X_train, Y_train,
                         verbose=params['keras_verbose_level'],
                         shuffle=params['lstm_shuffle'],
                         batch_size=params['lstm_batch_size'],
                         epochs=params['lstm_num_epochs'])
    return train_loss


def save(model):
    dt = datetime.now()
    name = '{0:%Y}{0:%m}{0:%d}_{0:%I}{0:%M}.h5'.format(dt)
    model.save(name)
    return name


def load(name):
    return load_model(name)
