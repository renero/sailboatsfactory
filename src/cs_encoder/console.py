from cs_encoder.oh_encoder import OHEncoder
from cs_encoder.cs_encoder import CSEncoder
from cs_encoder.ticks import Ticks
from cs_encoder.cs_nn import Csnn
from cs_encoder.dataset import Dataset
from cs_encoder.params import Params
# from cs_encoder.cs_plot import CSPlot

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
# cse_nn = Csnn().init('./nn_cse/params.yaml')
cse_bodies = Dataset().adjust(encoder.select_body(cse))
cse_shifts = Dataset().adjust(encoder.select_movement(cse))

#
# One hot encoding
#
oh_bodies = OHEncoder().fit(encoder.body_dict()).transform(cse_bodies)
oh_shifts = OHEncoder().fit(encoder.move_dict()).transform(cse_shifts)
body_sets = Dataset().train_test_split(data=oh_bodies)
move_sets = Dataset().train_test_split(data=oh_shifts)

nn_body = Csnn(body_sets)
# Load or build a model
# model = cse_nn.load_model('./nn_cse/networks/model_20180827_100_0.75')
model_body = nn_body.build_model()
nn_body.train(model_body)

nn_move = Csnn(move_sets)
# Load or build a model
# model = cse_nn.load_model('./nn_cse/networks/model_20180827_100_0.75')
model_move = nn_move.build_model()
nn_move.train(model_move)

#
# Reverse Encoding to produce ticks from CSE
#
cse_codes = encoder.read_cse(params._cse_file, params._cse_tags)
rec_ticks = encoder.cse2ticks(cse_codes.iloc[:params._n, ], params._ohlc_tags)
# -> CSPlot().plot(rec_ticks.iloc[:n, ], ohlc_names=ohlc_tags)
