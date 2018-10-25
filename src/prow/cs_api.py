import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from oh_encoder import OHEncoder
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


def prepare_datasets(encoder, cse, params):
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
    for name in params._names:
        call_select = getattr(encoder, 'select_{}'.format(name))
        cse_data[name] = Dataset().adjust(call_select(cse))
        oh_data[name] = encoder.onehot[name].encode(cse_data[name])
        dataset[name] = Dataset().train_test_split(oh_data[name])
    return dataset


def prepare_nn(dataset, params):
    """
    Make all the encodings and nn setup to train or load a recurrent network.
    The input are the ticks dataset, the encoder to be used and the list of
    encoded CSE, together with the params.
    :param ticks: the ticks file read
    :param encoder: the encoder used
    :param cse: the list of cse objects
    :param params: the parameters read from configuration file
    :return: the dictionary with the recurrent networks trained or load from
        file.
    """

    nn = {}
    for name in params._names:
        nn[name] = Csnn(name)
        if params._train is True:
            nn[name].build_model(dataset[name]).train().save()
        else:
            nn[name].load(params._model_filename[name], summary=False)
    return nn


def predict_testset(dataset, encoder, nn, params):
    """
    Run prediction for body and move over the testsets in the dataset object
    :param dataset: the data
    :param encoder:
    :param nn:
    :param params:
    :return:
    """
    prediction = {}
    for name in params._names:
        prediction[name] = Predict(dataset[name].X_test,
                                   dataset[name].y_test,
                                   encoder.onehot[name])
        call_predict = getattr(prediction[name],
                               'predict_{}_batch'.format(name))
        call_predict(nn[name])
    return prediction


def predict_next_close(ticks, encoder, nn, params):
    """
    From a list of ticks, make a prediction of what will be the next CS.
    :param ticks: a dataframe of ticks with the expected headers and size
    corresponding to the window size of the network to be used.
    :param encoder: the encoder used to train the network
    :param nn: the recurrent network to make the prediction with
    :param params: the parameters file read from configuration.
    :return: the close value of the CS predicted.
    """
    # encode the tick in CSE and OH. reshape it to the expected LSTM format.
    cs_tick = encoder.ticks2cse(ticks)
    cs_tick_body_oh = encoder.onehot['body'].encode(
        encoder.select_body(cs_tick))
    cs_tick_move_oh = encoder.onehot['move'].encode(
        encoder.select_move(cs_tick))
    input_body = cs_tick_body_oh.values.reshape((1, cs_tick_body_oh.shape[0],
                                                 cs_tick_body_oh.shape[1]))
    input_move = cs_tick_move_oh.values.reshape((1, cs_tick_move_oh.shape[0],
                                                 cs_tick_move_oh.shape[1]))

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

    # Convert the prediction to a real tick
    pred = encoder.cse2ticks(prediction_df, cs_tick[-1])
    return pred['c'].values[0]

    # Plot everything. The prediction is the last one. The actual is the
    # second last.
    # output_df = ticks.append(actual, ignore_index=True)
    # output_df = output_df.append(pred, ignore_index=True)
