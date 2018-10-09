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
ticks = Ticks().read_ohlc()
encoder = CSEncoder(log_level=params._LOG_LEVEL).fit(ticks, params._ohlc_tags)
cse = encoder.ticks2cse(ticks.iloc[:params._n, ])
encoder.save_cse(cse, params._cse_file)
# -> CSPlot().plot(ticks.iloc[:n, ], ohlc_names=ohlc_tags)

#
# Adjust dataset to fit into NN parameters
#
cse_bodies = Dataset().adjust(encoder.select_body(cse))
cse_shifts = Dataset().adjust(encoder.select_movement(cse))

#
# One hot encoding
#
oh_bodies = OHEncoder().fit(encoder.body_dict()).transform(cse_bodies)
oh_shifts = OHEncoder().fit(encoder.move_dict()).transform(cse_shifts)
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
        nn[i].load('./body_20181009_1503__100_0.528')
#
# Predict
#
test_bodies = encoder.select_body(cse[50:53])
test_oh_bodies = OHEncoder().fit(encoder.body_dict()).transform(test_bodies)
testset = test_oh_bodies.values.reshape((1, 3, 26))
y = nn[0].predict(testset)
plt.plot(y[0], '.-')
cse[54].info()
encoder.cse2ticks([cse[54]])
