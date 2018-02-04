from datetime import datetime
from keras.models import model_from_json
from os.path import join
from pathlib import Path


import lstm


def load_model(params, prediction=False):
    """
    Loads a model from file, and by default.
    The model is compiled according to the parameters specified
    in the 'params' dictionary.
    """
    home_path = str(Path.home())
    load_folder = join(join(home_path, params['project_path']),
                       params['networks_path'])
    if prediction is True:
        model_name = join(load_folder, params['pred_model'])
    else:
        model_name = join(load_folder, params['load_model'])
    # load json and create model
    json_file = open(model_name, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # Compile the model.
    loaded_model.compile(
        loss=params['lstm_loss'],
        optimizer=params['lstm_optimizer'])
    print('Model read from {}'.format(model_name))
    return loaded_model


def load_weights(model, params, prediction=False):
    """
    Load weights from existing .h5 file in 'networks_path' into existing
    compiled model
    """
    home_path = str(Path.home())
    load_folder = join(join(home_path, params['project_path']),
                       params['networks_path'])
    if prediction is True:
        net_name = join(load_folder, params['pred_weights'])
    else:
        net_name = join(load_folder, params['load_weights'])
    print("Weights read from {}".format(net_name))
    model.load_weights(net_name)
    return model


def setup(params):
    """
    If the parameters define a model to load, load it. Otherwise
    build it.
    """
    if 'load_model' in params and params['load_model']:
        model = load_model(params)
    else:
        model = lstm.build(params)
        print('Model built from scratch...')
    # load weights into new model if its defined and not empty.
    if 'load_weights' in params and params['load_weights']:
        model = load_weights(model, params)
    return model


def prediction_setup(params):
    model = load_model(params, prediction=True)
    return load_weights(model, params, prediction=True)


def save(model, params, prefix='', additional_epocs=0):
    """
    Save the model, weights and/or the predictor version of the net.
    """
    home_path = str(Path.home())
    save_folder = join(join(home_path, params['project_path']),
                       params['networks_path'])
    name_prefix = '{}_{:d}L{:d}u_{:d}e_{:02d}i_'.format(
        prefix,
        params['lstm_numlayers'],
        params['lstm_layer1'],
        params['lstm_num_epochs'] + additional_epocs,
        len(params['columNames']))
    name_suffix = '{0:%Y}{0:%m}{0:%d}_{0:%H}{0:%M}'.format(datetime.now())
    base_name = name_prefix + name_suffix
    # Save the model?
    if params['save_model']:
        model_name = join(save_folder, '{}.json'.format(base_name))
        model_json = model.to_json()
        with open(model_name, "w") as json_file:
            json_file.write(model_json)
        print('Model saved to {}'.format(model_name))
    # Save weights?
    if params['save_weights'] is True:
        net_name = join(save_folder, '{}.h5'.format(base_name))
        model.save_weights(net_name)
        print('Weights saved to {}'.format(net_name))
    # Save predictor?
    if params['save_predictor']:
        # Build the predictor model, with batch_size = 1, and save it.
        bs = params['lstm_batch_size']
        params['lstm_batch_size'] = 1
        pred_model = lstm.build(params)
        model_name = join(save_folder, '{}_pred.json'.format(base_name))
        model_json = pred_model.to_json()
        with open(model_name, "w") as json_file:
            json_file.write(model_json)
        params['lstm_batch_size'] = bs
        del pred_model
        print('1-Prediction model saved to {}'.format(model_name))
