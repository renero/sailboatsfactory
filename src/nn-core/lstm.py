from datetime import datetime
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Activation
from numpy.random import seed
from numpy import zeros
from tensorflow import set_random_seed
from os.path import join
from pathlib import Path


import compute
import data
import parameters


# Initialization of seeds
set_random_seed(2)
seed(2)


def build(params, save_predictor=True):
    """
    Build the LSTM according to the parameters passed. The general
    architecture is set in the code.
    """
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
    # model.add(Dense(input_dim=64, output_dim=1))  # <- this is under test.
    model.add(Dense(units=1, input_dim=64))  # <- this is under test.
    # model.add(Dense(params['lstm_predictions']))
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


def range_predict(model, X_test, Y_test, params):
    """
    Make a prediction for a range of input values, by saturating the lstm
    returns the predictions (unscaled) and the number of errors
    """
    input_shape = (1, params['lstm_timesteps'], len(params['columNames']))
    preds = zeros(X_test.shape[0])
    for i in range(0, X_test.shape[0]):
        input_vector = X_test[i].reshape(input_shape)
        # Make a prediction, saturating
        for k in range(0, params['lstm_timesteps']):
            y_hat = model.predict(input_vector,
                                  batch_size=params['lstm_batch_size'])
        preds[i] = y_hat
    rmse, num_errors = compute.error(Y_test, preds)
    return (preds, num_errors)


def predict(params_filename, net_name, dataset_name=''):
    """
    From a parameters file and a network name (model and weights), builds a
    prediction vector, which is returned together with the number of tendency
    errors found
    """
    params = parameters.read(params_filename)
    adjusted = parameters.adjust(data.read(params, dataset_name), params)
    _, _, X_test, Y_test = data.prepare(adjusted, params)
    model1 = load(net_name, params, prefix='pred_')
    (yhat, num_errors) = range_predict(model1, X_test, Y_test, params)
    return (Y_test, yhat, num_errors)


def save(model, name='', prefix='', save_weights=True):
    """
    Save the model and (by default) the weights too. If the parameter
    'save_weights' is set to False, only the model is saved.
    'prefix' is used to generate the name of the files, and is prepended
    to the data and extension to be used.
    """
    home_path = str(Path.home())
    project_path = 'Documents/SideProjects/sailboatsfactory'
    save_folder = join(join(home_path, project_path), 'data/networks')
    dt = datetime.now()
    if name is '':
        base_name = '{0:%Y}{0:%m}{0:%d}_{0:%H}{0:%M}'.format(dt)
    else:
        base_name = name
    net_name = join(save_folder, '{}.h5'.format(base_name))
    model_name = join(save_folder, (prefix + '{}.json'.format(base_name)))
    # Save the model
    model_json = model.to_json()
    with open(model_name, "w") as json_file:
        json_file.write(model_json)
    print('Model saved to {}'.format(model_name))
    # Save the weights
    if save_weights is True:
        print('Weights saved to {}'.format(net_name))
        model.save_weights(net_name)
    return (model_name, net_name)


def load(name, params, prefix='', load_weights=True):
    """
    Loads a model from file, and by default, also the weights.
    The model is compiled according to the parameters specified
    in the 'params' dictionary.
    """
    home_path = str(Path.home())
    load_folder = join(join(home_path, params['project_path']),
                       params['networks_path'])
    model_name = join(load_folder, (prefix + '{}.json'.format(name)))
    # load json and create model
    json_file = open(model_name, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    print('Model read from {}'.format(model_name))
    # load weights into new model
    if load_weights is True:
        net_name = join(load_folder, '{}.h5'.format(name))
        print("Weights read from {}".format(net_name))
        loaded_model.load_weights(net_name)
    loaded_model.compile(
        loss=params['lstm_loss'],
        optimizer=params['lstm_optimizer'])
    return loaded_model
