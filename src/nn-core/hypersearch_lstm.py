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

print('Results to output_{:d}.csv'.format(getpid()))
print('batch_size;timesteps;tst.error;tst.size;epochs',
      file=open('output_{:d}.txt'.format(getpid()), "a"))
params = parameters.read()
raw = data.read(params)
print('Original dataset num samples:', raw.shape)

# Search through the space of batch_size and timesteps the best
# trend error. Results are dumped to file output_PID.txt
for bs in [8, 10, 12, 14, 16]:
    for ts in [8, 10, 12, 14, 16]:
        params['lstm_batch_size'] = bs
        params['lstm_timesteps'] = ts
        adjusted = parameters.adjust(raw, params)
        X_train, Y_train, X_test, Y_test = data.prepare(adjusted, params)
        model = lstm.build(params)
        train_loss = lstm.fit(model, X_train, Y_train, params)
        Y_hat = model.predict(X_test, batch_size=params['lstm_batch_size'])
        rmse, num_errors = compute.error(Y_test, Y_hat)
        te = (num_errors / (len(Y_hat)-1))
        print('{:d};{:d};{:.02f};{:d};{:d}'
              .format(bs, ts, te, (len(Y_hat)-1), params['lstm_num_epochs']),
              file=open('output_{:d}.txt'.format(getpid()), "a"))
        if te <= 0.15:
            name = lstm.save(model)
            print("Saved model:", name)
        del model
