from pandas import read_csv
from sklearn.model_selection import train_test_split

from candle import Candle


def read_ticks(filename):
    content = read_csv(filename,
                       sep='|', header='infer', delimiter='|', engine='python',
                       usecols=['tickcloseprice',
                                'tickopenprice',
                                'tickminprice',
                                'tickmaxprice'])
    return content


# if len(sys.argv) != 2:
#     print('Usage: python {} filename.csv'.format(sys.argv[0]))
#     sys.exit(1)
datafile = '/Users/renero/Documents/SideProjects/sailboatsfactory/data/100.csv'
# A sliding window of 'k', that will contain the
# k-n elements of a sequence, and the expected 'n' values', as predictions.
k = 4
n = 1

raw = read_ticks(datafile)
training, test = train_test_split(raw, test_size=0.2)

# This class allows me to encode the candlesticks, so I do it for every row.
# I must pass all raw values to compute max and min values.
candle = Candle(raw)

# Encode every row, using a lambda function.
# The last k elements of the sliding window contain NaNs, as the sliding window
# doesn't work for them.
training_candles = training.apply(lambda row: candle.encode(row),
                                  axis=1).to_frame()
training_sliding = candle.make_sliding(training_candles, k).iloc[0:-k, :]

# And now, repeat it for the test set
test_candles = test.apply(lambda row: candle.encode(row),
                          axis=1).to_frame()
test_sliding = candle.make_sliding(test_candles, k-n).iloc[0:-k, :]

# Save everything to CSV files.
training_sliding.to_csv('training.csv', header=False, index=False, sep='|')
test_sliding.to_csv('test.csv', header=False, index=False, sep='|')
