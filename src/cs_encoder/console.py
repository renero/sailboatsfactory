from cs_encoder.oh_encoder import OHEncoder
from cs_encoder.cs_encoder import CSEncoder
from cs_encoder.ticks import Ticks
from cs_encoder.cs_nn import Csnn
from cs_encoder.dataset import Dataset
from cs_encoder.params import Params

#
# Read raw data, and encode it.
#
params = Params()
if params._cse_file is not None:
    params.log.info('Reading and encoding ticksfile: {}'.format(
        params._ticks_file))
    ticks = Ticks().read_ohlc()
    encoder = CSEncoder().fit(ticks, params._ohlc_tags)
    cse = encoder.ticks2cse(ticks.iloc[:params._n, ])
    encoder.save_cse(cse, params._cse_file)
    # -> CSPlot().plot(ticks.iloc[:n, ], ohlc_names=ohlc_tags)
else:
    params.log.info('Reading CSE from file: {}'.format(params._cse_file))
    encoder = CSEncoder()
    cse = encoder.read_cse()

#
# Adjust dataset to fit into NN parameters
#
cse_bodies = Dataset().adjust(encoder.select_body(cse))
cse_shifts = Dataset().adjust(encoder.select_movement(cse))

#
# One hot encoding
#
oh_encoder_body = OHEncoder().fit(encoder.body_dict())
oh_encoder_move = OHEncoder().fit(encoder.move_dict())
oh_bodies = oh_encoder_body.transform(cse_bodies)
oh_shifts = oh_encoder_move.transform(cse_shifts)
body_sets = Dataset().train_test_split(data=oh_bodies)
move_sets = Dataset().train_test_split(data=oh_shifts)

#
# Load or build a model
#
nn = []
data = [move_sets, body_sets]
for i, model_name in enumerate(['move']):  # , 'body']):
    nn.append(Csnn(data[i], model_name))

if params._train is True:
    for i in range(len(nn)):
        nn[i].build_model()
        nn[i].train()
        nn[i].save()
else:
    for i in range(len(nn)):
        nn[i].load(params._model_filename[i], summary=False)
#
# Predict Body
#
positive_all = 0
positive_sign = 0
positive_shape = 0
nn_index = 1
num_testcases = body_sets.X_test.shape[0]
for j in range((body_sets.X_test.shape[0]) - 2):
    y = nn[nn_index].predict(body_sets.X_test[j:j + 1, :, :])
    y_pred = nn[nn_index].hardmax(y[0])
    cse_predicted = oh_encoder_body.decode(y_pred)[0]
    cse_actual = oh_encoder_body.decode(body_sets.y_test[j:j + 1, :])[0]
    positive_all += int(cse_actual == cse_predicted)
    positive_sign += int(cse_actual[0] == cse_predicted[0])
    positive_shape += int(cse_actual[-1] == cse_predicted[-1])
    # print('predicted: {} / actual: {}'.format(cse_actual))

print('Pos.Rate (all/sign/body): {:.3f} / {:.3f} / {:.3f}'.format(
    (positive_all / num_testcases), (positive_sign / num_testcases),
    (positive_shape / num_testcases)))

#
# Predict Move
#
pos_open = 0
pos_close = 0
pos_high = 0
pos_low = 0
nn_index = 0
pred_length = len(oh_encoder_move._states)
num_predictions = int(move_sets.y_test.shape[1] / pred_length)
j = 0

for j in range((move_sets.X_test.shape[0]) - 2):
    y = nn[nn_index].predict(move_sets.X_test[j:j + 1, :, :])
    Y_pred = [
        nn[nn_index].hardmax(
            y[0][i * pred_length:(i * pred_length) + pred_length - 1])
        for i in range(num_predictions)
    ]
    move_predicted = [
        oh_encoder_move.decode(Y_pred[i])[0] for i in range(num_predictions)
    ]
    move_actual = [
        oh_encoder_move.decode(move_sets.y_test[j:j + 1, :])[0]
        for i in range(num_predictions)
    ]
    pos_open += int(move_actual[0] == move_predicted[0])
    pos_high += int(move_actual[1] == move_predicted[1])
    pos_low += int(move_actual[2] == move_predicted[2])
    pos_close += int(move_actual[3] == move_predicted[3])

num_testcases = (move_sets.X_test.shape[0]) - 2
print('Pos.Rate (O/H/L/C): {:.4f} : {:.4f} : {:.4f} : {:.4f} ~ {:.4f}'.
      format((pos_open / num_testcases),
             (pos_high / num_testcases),
             (pos_low / num_testcases),
             (pos_close / num_testcases),
             ((pos_open+pos_high+pos_low+pos_close)/(num_testcases*4))))
