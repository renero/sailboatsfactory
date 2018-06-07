from numpy.random import seed
from tensorflow import set_random_seed
import data
import lstm
import model
import parameters
import plot


%load_ext autoreload
%autoreload 2

set_random_seed(2)
seed(2)

#
# s e t u p
#
raw, params = parameters.initialize()
normalized = data.normalize(raw, params)
parameters.summary(params)
X, Y, Xtest, ytest = data.prepare(normalized, params)

#
# t r a i n i n g
#
model = model.setup(params)
model.summary()
lstm.stateless_fit(model, X, Y, Xtest, ytest, params)
# model.save(model, params, prefix='5y', additional_epocs=0)

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
