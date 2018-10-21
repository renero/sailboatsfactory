import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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


def predict_next_close(ticks, cs_encoder, oh_encoder, nn, params):
    # encode the tick in CSE and OH. reshape it to the expected LSTM format.
    cs_tick = cs_encoder.ticks2cse(ticks)
    cs_tick_body_oh = oh_encoder['body'].encode(
        cs_encoder.select_body(cs_tick))
    cs_tick_move_oh = oh_encoder['move'].encode(
        cs_encoder.select_move(cs_tick))
    input_body = cs_tick_body_oh.values.reshape((1, cs_tick_body_oh.shape[0],
                                                 cs_tick_body_oh.shape[1]))
    input_move = cs_tick_move_oh.values.reshape((1, cs_tick_move_oh.shape[0],
                                                 cs_tick_move_oh.shape[1]))

    # get a prediction from the proper networks, for the body part
    raw_prediction = nn['body'].predict(input_body)[0]
    pred_body_oh = nn['body'].hardmax(raw_prediction)
    pred_body_cs = oh_encoder['body'].decode(pred_body_oh)

    # Repeat everything with the move:
    # get a prediction from the proper network, for the MOVE part
    pred_length = len(oh_encoder['move']._states)
    num_predictions = int(input_move.shape[2] / pred_length)
    y = nn['move'].predict(input_move)[0]
    Y_pred = [
        nn['move'].hardmax(y[i * pred_length:(i * pred_length) + pred_length])
        for i in range(num_predictions)
    ]
    pred_move_cs = [
        oh_encoder['move'].decode(Y_pred[i])[0] for i in range(num_predictions)
    ]

    # Decode the prediction into a normal tick (I'm here!!!)
    prediction_df = pd.DataFrame([], columns=params._cse_tags)
    prediction_cs = np.concatenate((pred_body_cs, pred_move_cs), axis=0)
    this_prediction = dict(zip(params._cse_tags, prediction_cs))
    prediction_df = prediction_df.append(this_prediction, ignore_index=True)

    # Convert the prediction to a real tick
    pred = cs_encoder.cse2ticks(prediction_df, cs_tick[-1])
    return pred['c'].values[0]

    # Plot everything. The prediction is the last one. The actual is the
    # second last.
    # output_df = ticks.append(actual, ignore_index=True)
    # output_df = output_df.append(pred, ignore_index=True)
