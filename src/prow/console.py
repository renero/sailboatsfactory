from oh_encoder import OHEncoder
from cs_encoder import CSEncoder
from ticks import Ticks
from cs_nn import Csnn
from dataset import Dataset
from params import Params
from predict import Predict

import numpy as np
import pandas as pd

params = Params()
ticks = Ticks().read_ohlc()
encoder = CSEncoder().fit(ticks, params._ohlc_tags)
cse = encoder.ticks2cse(ticks)

oh_encoder = {}
cse_data = {}
oh_data = {}
dataset = {}
nn = {}
prediction = {}

for name in params._names:
    call_select = getattr(encoder, 'select_{}'.format(name))
    cse_data[name] = Dataset().adjust(call_select(cse))

    call_dict = getattr(encoder, '{}_dict'.format(name))
    oh_encoder[name] = OHEncoder().fit(call_dict())

    oh_data[name] = oh_encoder[name].encode(cse_data[name])
    dataset[name] = Dataset().train_test_split(oh_data[name])

    nn[name] = Csnn(dataset[name], name)

    if params._train is True:
        nn[name].build_model().train().save()
    else:
        nn[name].load(params._model_filename[name], summary=False)

    prediction[name] = Predict(dataset[name].X_test, dataset[name].y_test,
                               oh_encoder[name])
    call_predict = getattr(prediction[name], '{}_batch'.format(name))
    call_predict(nn[name])

#
#
#
#
# Crazy one tick conversion.......
#
#
#
#
# Single prediction case
pos = 10
# select tick and prediction. This must come in clear from the API or NOT!
tick = ticks.iloc[pos:pos + params._window_size]
actual = ticks.iloc[pos + params._window_size + 1:pos + params._window_size +
                    2]

# encode the tick in CSE and OH. reshape it to the expected LSTM format.
cs_tick = encoder.ticks2cse(tick)
cs_tick_body_oh = oh_encoder['body'].encode(encoder.select_body(cs_tick))
cs_tick_move_oh = oh_encoder['move'].encode(encoder.select_move(cs_tick))
input_body = cs_tick_body_oh.values.reshape((1, cs_tick_body_oh.shape[0],
                                             cs_tick_body_oh.shape[1]))
input_move = cs_tick_move_oh.values.reshape((1, cs_tick_move_oh.shape[0],
                                             cs_tick_move_oh.shape[1]))

# do the same with the prediction, in CSE format.
cs_actual = encoder.ticks2cse(actual)
cs_actual_body_oh = oh_encoder['body'].encode(encoder.select_body(cs_actual))
cs_actual_move_oh = oh_encoder['move'].encode(encoder.select_move(cs_actual))

# get a prediction from the proper networks, for the body part
pred_body_oh = nn['body'].hardmax(nn['body'].predict(input_body)[0])
pred_body_cs = oh_encoder['body'].decode(pred_body_oh)

# Repeat everything with the move:
# get a prediction from the proper network, for the MOVE part
y = nn['move'].predict(input_move)[0]
pred_length = len(oh_encoder['move']._states)
num_predictions = int(input_move.shape[2] / pred_length)
Y_pred = [
    nn['move'].hardmax(y[i * pred_length:(i * pred_length) + pred_length])
    for i in range(num_predictions)
]
pred_move_cs = [
    oh_encoder['move'].decode(Y_pred[i])[0] for i in range(num_predictions)
]

# Decode the prediction into a normal tick (I'm here!!!)
prediction_cs = np.concatenate((pred_body_cs, pred_move_cs), axis=0)
prediction_df = pd.DataFrame([], columns=params._cse_tags)
prediction_df.append(
    dict(zip(params._cse_tags, prediction_cs)), ignore_index=True)

encoder.cse2ticks(prediction_df)
actual
