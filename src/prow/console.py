from oh_encoder import OHEncoder
from cs_encoder import CSEncoder
from ticks import Ticks
from cs_nn import Csnn
from dataset import Dataset
from params import Params
from predict import Predict


params = Params()
ticks = Ticks().read_ohlc()
encoder = CSEncoder().fit(ticks, params._ohlc_tags)
cse = encoder.ticks2cse(ticks)
encoder.save_cse(cse, params._cse_file)

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

    oh_data[name] = oh_encoder[name].transform(cse_data[name])
    dataset[name] = Dataset().train_test_split(oh_data[name])

    nn[name] = Csnn(dataset[name], name)

    if params._train is True:
        nn[name].build_model().train().save()
    else:
        nn[name].load(params._model_filename[name], summary=False)

    prediction[name] = Predict(dataset[name], oh_encoder[name])
    call_predict = getattr(prediction[name], name)
    call_predict(nn[name])
