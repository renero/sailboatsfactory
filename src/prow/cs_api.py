import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cs_nn import Csnn
from dataset import Dataset
from predict import Predict


def plot_body_prediction(raw_prediction, pred_body_cs):
    # Plot the raw prediction from the NN
    fig, ax = plt.subplots(figsize=(6, 4))
    plt.plot(raw_prediction)
    winner_prediction = max(raw_prediction, key=abs)
    pos = np.where(raw_prediction == winner_prediction)[0][0]
    plt.plot(pos, winner_prediction, 'yo')
    plt.annotate(
        '{}={}'.format(pos, pred_body_cs[0]),
        xy=(pos, winner_prediction),
        xytext=(pos + 0.5, winner_prediction))
    plt.xticks(np.arange(0, len(raw_prediction), 1.0))
    ax.xaxis.label.set_size(6)


def plot_move_prediction(y, Y_pred, pred_move_cs, num_predictions,
                         pred_length):
    # find the position of the absmax mvalue in each of the arrays
    y_maxed = np.zeros(y.shape)
    for i in range(num_predictions):
        winner_prediction = max(Y_pred[i], key=abs)
        pos = np.where(Y_pred[i] == winner_prediction)[0][0]
        y_maxed[(i * pred_length) + pos] = winner_prediction

    # Plot the raw prediction from the NN
    fig, ax = plt.subplots(figsize=(6, 4))
    plt.plot(y)
    for i in range(len(y_maxed)):
        if y_maxed[i] != 0.0:
            plt.plot(i, y[i], 'yo')
            plt.annotate(
                '{}={}'.format(i, pred_move_cs[int(i / pred_length)]),
                xy=(i, y[i]),
                xytext=(i + 0.6, y[i]))
    plt.xticks(np.arange(0, len(y), 2.0))
    ax.xaxis.label.set_size(2)
    for vl in [i * pred_length for i in range(num_predictions + 1)]:
        plt.axvline(x=vl, linestyle=':', color='red')


def prepare_datasets(encoder, cse, subtypes):
    """
    Prepare the training and test datasets from an list of existing CSE, for
    each of the model names considered (body and move).

    :param encoder: The encoder used to build the CSE list.
    :param cse: The list of CSE objects
    :param params: The parameters read from file
    :return: The datasets for each of the models that need to be built. The
        names of the models specify the 'body' part and the 'move' part.
    """
    cse_data = {}
    oh_data = {}
    dataset = {}
    for subtype in subtypes:
        call_select = getattr(encoder, '{}'.format(subtype))
        cse_data[subtype] = Dataset().adjust(call_select(cse))
        oh_data[subtype] = encoder.onehot[subtype].encode(cse_data[subtype])
        dataset[subtype] = Dataset().train_test_split(oh_data[subtype])
    return dataset


def train_nn(dataset, subtypes):
    """
    Train a model.
    """
    nn = {}
    for subtype in subtypes:
        nn[subtype] = Csnn(None, subtype)

        window_size = dataset[subtype].X_train.shape[1]
        num_categories = dataset[subtype].X_train.shape[2]

        nn[subtype].build_model(window_size, num_categories).train(
            dataset[subtype].X_train, dataset[subtype].y_train).save()

    return nn


def load_nn(model_names, subtypes):
    """
    """
    nn = {}
    for name in model_names.keys():
        nn[name] = {}
        for subtype in subtypes:
            nn[name][subtype] = Csnn(name, subtype)
            nn[name][subtype].load(model_names[name][subtype])
    return nn


def predict_testset(dataset, encoder, nn, subtypes):
    """
    Run prediction for body and move over the testsets in the dataset object
    :param dataset: the data
    :param encoder:
    :param nn:
    :param params:
    :return:
    """
    prediction = {}
    for name in subtypes:
        prediction[name] = Predict(dataset[name].X_test,
                                   dataset[name].y_test,
                                   encoder.onehot[name])
        call_predict = getattr(prediction[name],
                               'predict_{}_batch'.format(name))
        call_predict(nn[name])
    return prediction


def predict_close(ticks, encoder, nn, params):
    """
    From a list of ticks, make a prediction of what will be the next CS.
    :param ticks: a dataframe of ticks with the expected headers and size
    corresponding to the window size of the network to be used.
    :param encoder: the encoder used to train the network
    :param nn: the recurrent network to make the prediction with
    :param params: the parameters file read from configuration.
    :return: the close value of the CS predicted.
    """
    # Check that the input group of ticks match the size of the window of
    # the network that is going to make the predict. That parameter is in
    # the window_size attribute within the 'encoder'.
    if ticks.shape[0] != encoder.window_size():
        info_msg = 'Tickgroup resizing: {} -> {}'
        params.log.info(info_msg.format(ticks.shape[0], encoder.window_size()))
        ticks = ticks.iloc[-encoder.window_size():, :]
        ticks.reset_index()

    # encode the tick in CSE and OH. Reshape it to the expected LSTM format.
    cs_tick = encoder.ticks2cse(ticks)
    cs_tick_body_oh = encoder.onehot['body'].encode(encoder.body(cs_tick))
    cs_tick_move_oh = encoder.onehot['move'].encode(encoder.move(cs_tick))

    input_body = cs_tick_body_oh.values[np.newaxis, :, :]
    input_move = cs_tick_move_oh.values[np.newaxis, :, :]

    # get a prediction from the proper networks, for the body part
    raw_prediction = nn['body'].predict(input_body)[0]
    pred_body_oh = nn['body'].hardmax(raw_prediction)
    pred_body_cs = encoder.onehot['body'].decode(pred_body_oh)

    # Repeat everything with the move:
    # get a prediction from the proper network, for the MOVE part
    pred_length = len(encoder.onehot['move']._states)
    num_predictions = int(input_move.shape[2] / pred_length)
    y = nn['move'].predict(input_move)[0]
    Y_pred = [
        nn['move'].hardmax(y[i * pred_length:(i * pred_length) + pred_length])
        for i in range(num_predictions)
    ]
    pred_move_cs = [
        encoder.onehot['move'].decode(Y_pred[i])[0] for i in
        range(num_predictions)
    ]

    # Decode the prediction into a normal tick (I'm here!!!)
    prediction_df = pd.DataFrame([], columns=params._cse_tags)
    prediction_cs = np.concatenate((pred_body_cs, pred_move_cs), axis=0)
    this_prediction = dict(zip(params._cse_tags, prediction_cs))
    prediction_df = prediction_df.append(this_prediction, ignore_index=True)
    params.log.info('Net {} ID 0x{} -> {}:{}|{}|{}|{}'.format(
        nn['body'].name,
        hex(id(nn)),
        prediction_df[params._cse_tags[0]].values[0],
        prediction_df[params._cse_tags[1]].values[0],
        prediction_df[params._cse_tags[2]].values[0],
        prediction_df[params._cse_tags[3]].values[0],
        prediction_df[params._cse_tags[4]].values[0],
    ))

    # Convert the prediction to a real tick
    pred = encoder.cse2ticks(prediction_df, cs_tick[-1])
    return pred['c'].values[0]

    # Plot everything. The prediction is the last one. The actual is the
    # second last.
    # output_df = ticks.append(actual, ignore_index=True)
    # output_df = output_df.append(pred, ignore_index=True)

#
# Single prediction case, in sequence mode.
# errors = []
# for i in range(10):
#     start = 20 + i
#     end = start + params._window_size
#     tick_group = ticks.iloc[start:end]
#     real_close = ticks.iloc[end:end + 1]['c'].values[0]
#     for name in params._model_names:
#         nn_encoder = CSEncoder().load(params._model_names[name]['encoder'])
#         next_close = predict_close(tick_group, nn_encoder, nn[name], params)
#         errors.append(abs(next_close - real_close))
#
# plt.plot(errors)
# med = np.median(errors)
# std = np.std(errors)
# plt.axhline(med, linestyle=':', color='red')
# plt.axhline(med + std, linestyle=':', color='green')
# plt.show()
# plt.hist(errors, color='blue', edgecolor='black', bins=int(100 / 2))
# plt.show()
