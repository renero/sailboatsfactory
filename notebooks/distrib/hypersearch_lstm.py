from os import getpid
from numpy.random import seed
from tensorflow import set_random_seed

import compute
import data
import lstm
import parameters


# Initialization of seeds
set_random_seed(2)
seed(2)

params = parameters.read()
raw = data.read(params)
print('Original dataset num samples:', raw.shape)

# Search through the space of batch_size and timesteps the best
# trend error. Results are dumped to file output_PID.txt
for bs in [1, 2, 4, 6, 8]:
    for ts in [1, 2, 4, 6, 8]:
        params['lstm_batch_size'] = bs
        params['lstm_timesteps'] = ts
        adjusted = parameters.adjust(raw, params)
        X_train, Y_train, X_test, Y_test = data.prepare(adjusted, params)
        model = lstm.build(params)
        train_loss = model.fit(
            X_train, Y_train,
            shuffle=params['lstm_shuffle'],
            batch_size=params['lstm_batch_size'],
            epochs=params['lstm_num_epochs'])
        Y_hat = model.predict(X_test, batch_size=params['lstm_batch_size'])
        rmse, trend_error = compute.error(Y_test, Y_hat)
        print('bs:{:d}, ts:{:d}, t.e.:{:.02f}, epochs:{:d}'
              .format(ts, bs, trend_error, params['lstm_num_epochs']),
              file=open('output_{:d}.txt'.format(getpid()), "a"))
