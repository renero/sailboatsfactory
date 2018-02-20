import matplotlib.pyplot as plt
from numpy.random import seed
from numpy import log
from tensorflow import set_random_seed
from keras.models import clone_model
from keras.callbacks import EarlyStopping
import compute
from data import read, normalize, denormalize, prepare
import lstm
from model import setup, save
import parameters
import plot


# %matplotlib inline
%load_ext autoreload
%autoreload 2
# Initialization of seeds
set_random_seed(2)
seed(2)

#
# s e t u p
#
params = parameters.read()
adjusted = parameters.adjust(read(params), params)
X, Y, Xtest, ytest = prepare(normalize(adjusted, params), params)

#
# t r a i n i n g
#
model = setup(params)
parameters.summary(params)
model.summary()
lstm.stateless_fit(model, X, Y, Xtest, ytest, params)
EarlyStopping(
    monitor='val_loss', min_delta=0.0,
    patience=0, verbose=0, mode='auto')
save(model, params, prefix='5y', additional_epocs=0)

#
# r e b u i l d   &   p r e d i c t
#
pred = lstm.build(params, batch_size=1)
pred.set_weights(model.get_weights())
(yhat, rmse, num_errors) = lstm.range_predict(pred, Xtest, ytest, params)

#
# p l o t
#
# plot.history(train_loss)
plot.prediction(ytest, yhat, rmse, num_errors, params)
