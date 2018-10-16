from cs_encoder.oh_encoder import OHEncoder
from cs_encoder.cs_encoder import CSEncoder
from cs_encoder.ticks import Ticks
from cs_encoder.cs_nn import Csnn
from cs_encoder.dataset import Dataset
from cs_encoder.params import Params
from cs_encoder.predict import Predict

#
# Read raw data, and encode it.
#
params = Params()
params.log.info('Reading and encoding ticksfile: {}'.format(
    params._ticks_file))
ticks = Ticks().read_ohlc()
encoder = CSEncoder().fit(ticks, params._ohlc_tags)
cse = encoder.ticks2cse(ticks.iloc[:params._n, ])
encoder.save_cse(cse, params._cse_file)

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
for i, model_name in enumerate(params._model_filename):
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
# Make batch predictions
#
Predict(body_sets, oh_encoder_body).body(nn[0])
Predict(move_sets, oh_encoder_move).move(nn[1])
