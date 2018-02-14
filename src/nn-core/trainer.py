import matplotlib.pyplot as plt
from numpy.random import seed
from numpy import log
from tensorflow import set_random_seed
import compute
import data
import lstm
from model import setup, save, latest_network
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
raw = data.read(params)
adjusted = parameters.adjust(raw, params)
normalized = data.normalize(adjusted, params)
X_train, Y_train, X_test, Y_test = data.prepare(normalized, params)

#
# t r a i n i n g
#
model = setup(params)
train_loss = lstm.fit(model, X_train, Y_train, params)
save(model, params, prefix='5y', additional_epocs=0)

#
# p r e d i c t i n g
#
params['pred_model'] = latest_network(params)
params['pred_weights'] = '{}.h5'.format(params['pred_model'][:-10])
(params, _, y, yhat, num_errors) = lstm.predict(params)
plot.prediction(y, yhat, num_errors, params)

#
#
#
#
# d e b u g
#
#
#
#

# plt.style.use('dark_background')
# plt.plot(raw)
# plt.xlim(0, 100)
# plt.show()
# plt.plot(Y_train)
# plt.xlim(0, 100)
# plt.show()
