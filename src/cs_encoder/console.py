from cs_encoder.oh_encoder import OHEncoder
from cs_encoder.cs_encoder import CSEncoder
from cs_encoder.ticks import Ticks
from cs_encoder.cs_nn import Csnn
from cs_encoder.dataset import Dataset
from cs_encoder.params import Params
# from cs_encoder.cs_plot import CSPlot

import matplotlib.pyplot as plt

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
data = [body_sets, move_sets]
for i, model_name in enumerate(['body']):
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
# Predict
#
positive_all = 0
positive_sign = 0
positive_shape = 0
num_predictions = body_sets.X_test.shape[0]
for j in range(body_sets.X_test.shape[0]):
    y = nn[0].predict(body_sets.X_test[j:j + 1, :, :])
    y_pred = nn[0].hardmax(y)
    cse_predicted = oh_encoder_body.decode(y_pred)[0]
    cse_actual = oh_encoder_body.decode(body_sets.y_test[j:j + 1, :])[0]
    positive_all += int(cse_actual == cse_predicted)
    positive_sign += int(cse_actual[0] == cse_predicted[0])
    positive_shape += int(cse_actual[-1] == cse_predicted[-1])
    print('predicted: {} / actual: {}'.format(cse_actual))
print('PR (all): {:.3f}\nPR (sign): {:.3f}\nPR (shape): {:.3f}'.format(
    (positive_all / num_predictions),
    (positive_sign / num_predictions),
    (positive_shape / num_predictions)))
body_sets.y_test[j:j + 1, :].shape
