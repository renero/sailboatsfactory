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
cse = encoder.ticks2cse(ticks)
encoder.save_cse(cse, params._cse_file)

#
# One hot encoding
#
oh_encoder = {}
cse_data = {}
oh_data = {}
dataset = {}
dataobj = {}
nn = {}
predict = {}
for name in params._names:
    method = getattr(encoder, 'select_{}'.format(name))
    cse_data[name] = Dataset().adjust(method(cse))
    method = getattr(encoder, '{}_dict'.format(name))
    oh_encoder[name] = OHEncoder().fit(method())
    oh_data[name] = oh_encoder[name].transform(cse_data[name])
    dataset[name] = Dataset().train_test_split(oh_data[name])
    nn[name] = Csnn(dataset[name], name)
    if params._train is True:
        nn[name].build_model()
        nn[name].train()
        nn[name].save()
    else:
        nn[name].load(params._model_filename[name], summary=False)
    predict[name] = Predict(dataset[name], oh_encoder[name])
    method = getattr(predict, name)
    predict.method(nn[name])

# cse_body = Dataset().adjust(encoder.select_body(cse))
# cse_move = Dataset().adjust(encoder.select_move(cse))
# oh_encoder_body = OHEncoder().fit(encoder.body_dict())
# oh_encoder_move = OHEncoder().fit(encoder.move_dict())
# oh_bodies = oh_encoder_body.transform(cse_body)
# oh_shifts = oh_encoder_move.transform(cse_move)
# body_sets = Dataset().train_test_split(data=oh_bodies)
# move_sets = Dataset().train_test_split(data=oh_shifts)
# nn = {}
# for name in params._nn_names:
#     nn[name] = Csnn(dataset[name], name)
# if params._train is True:
#     for i in range(len(nn)):
#         nn[i].build_model()
#         nn[i].train()
#         nn[i].save()
#         gc.collect()
# else:
#     for i in range(len(nn)):
#         nn[i].load(params._model_filename[i], summary=False)
#
# Make batch predictions
#
# Predict(body_sets, oh_encoder_body).body(nn[0])
# Predict(move_sets, oh_encoder_move).move(nn[1])
