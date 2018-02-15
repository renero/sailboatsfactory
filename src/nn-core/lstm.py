from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Activation
from numpy.random import seed
from numpy import zeros
from tensorflow import set_random_seed

import compute
import data
import model
import parameters


# Initialization of seeds
set_random_seed(2)
seed(2)


def build(params, batch_size=None):
    """
    Build the LSTM according to the parameters passed. The general
    architecture is set in the code.
    :param batch_size: If this param is not None is used to override the value
                       set in the parameters dictionary. This is usefule when
                       willing to build a network to make 1-step predictions.
    """
    # Use ALWAYS the batch_size value from the parameter of the method. If not
    # set, then copy it from the params.
    if batch_size is None:
        batch_size = params['lstm_batch_size']
    # Buuild the lstm.
    model = Sequential()
    # Check if my design has more than 1 layer.
    ret_seq_flag = False
    if params['lstm_numlayers'] > 1:
        ret_seq_flag = True
    # Add input layer.
    print('Adding layer #{:d} [{:d}]'
          .format(1, params['lstm_layer{:d}'.format(1)]))
    model.add(LSTM(
            params['lstm_layer1'],
            stateful=params['lstm_stateful'],
            unit_forget_bias=params['lstm_forget_bias'],
            unroll=params['lstm_unroll'],
            batch_input_shape=(batch_size,
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
    model.add(Dense(units=1, input_dim=params['lstm_layer{:d}'.format(
        params['lstm_numlayers'])]))
    model.add(Activation('linear'))
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


def single_predict(model, x_test, y_test, params):
    """
    Make a prediction for a single of input values, by saturating the lstm
    returns the prediction, unscaled.
    """
    # Takes the input vector, to make a prediction.
    input_shape = (1, params['lstm_timesteps'], len(params['columNames']))
    input_vector = x_test.reshape(input_shape)
    # Make a prediction by repeating the process 'n' times (n ~ timesteps.)
    for k in range(0, params['lstm_timesteps']):
        y_hat = model.predict(input_vector, batch_size=1)
    return y_hat


def range_predict(model, X_test, Y_test, params, batch_size=1):
    """
    Make a prediction for a range of input values, by saturating the lstm
    returns the predictions (unscaled) and the number of errors
    """
    input_shape = (1, params['lstm_timesteps'], len(params['columNames']))
    preds = zeros(X_test.shape[0])
    for i in range(0, X_test.shape[0]):
        input_vector = X_test[i].reshape(input_shape)
        # Make a prediction, saturating
        for k in range(0, params['num_saturations']):
            y_hat = model.predict(input_vector, batch_size=batch_size)
        preds[i] = y_hat
    rmse, num_errors = compute.error(Y_test, preds)
    return (preds, rmse, num_errors)


def predict(params):
    """
    From a set of parameters, loads a network (model and weights), builds a
    prediction vector, which is returned together with the number of tendency
    errors found
    """
    raw = data.read(params, params['pred_dataset'])
    normalized = data.normalize(raw, params)
    adjusted = parameters.adjust(normalized, params)
    # prepare test data
    _, _, X_test, Y_test = data.prepare(adjusted, params)
    # Perform the prediction.
    model1 = model.prediction_setup(params)
    print('Feeding X_test (shape=', X_test.shape, ')')
    (yhat, rmse, num_errors) = range_predict(model1, X_test, Y_test, params)
    return (params, model1, Y_test, yhat, rmse, num_errors)
