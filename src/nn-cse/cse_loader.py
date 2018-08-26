import pickle
import pandas as pd
import numpy as np

from keras.layers import LSTM, Dense, Dropout, Activation
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


%matplotlib inline
num_categories = 52
window_size = 3
_dropout = 0.1


def read_file():
    """Read the file and return a Series object with a column called 'cse'"""
    f = pd.read_csv('./result.csv', 'r', header='infer', delimiter=',')
    return f.cse


def build_encoding_equation(n, min, max):
    x1 = 0
    x2 = n - 1
    y1 = min
    y2 = max
    m = (y2 - y1) / (x2 - x1)
    b = y1 - (m * x1)
    return lambda x: float('{0:.2f}'.format(((m * x) + b)))


def build_dictionary(data, regression=False):
    unique_values = sorted(data.unique())
    if regression is True:
        eq = build_encoding_equation(len(unique_values), -1.0, +1.0)
        dictionary = {
            value: eq(idx)
            for (idx, value) in enumerate(unique_values)
        }
    else:
        dictionary = {value: idx for (idx, value) in enumerate(unique_values)}
    return dictionary


def encode(raw_data, dictionary):
    return raw_data.apply(lambda v: dictionary[v])


def to_numerical(raw_data):
    dictionary = build_dictionary(raw_data)
    df = raw_data.apply(lambda ev: dictionary[ev])
    return df


# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
    df = pd.DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag + 1)]
    columns.append(df)
    df = pd.concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return train_test_split(df, test_size=0.1, shuffle=False)


def reshape(data):
    num_entries = data.shape[0] * data.shape[1]
    timesteps = window_size + 1
    num_samples = int((num_entries / num_categories) / timesteps)
    train = data.reshape((num_samples, timesteps, num_categories))
    X_train = train[:, 0:window_size, :]
    y_train = train[:, -1, :]
    return X_train, y_train


def to_slidingwindow_series(data, window_size, test_size):
    series = data.copy()
    series_s = series.copy()
    for i in range(window_size):
        series = pd.concat([series, series_s.shift(-(i + 1))], axis=1)
    series.dropna(axis=0, inplace=True)
    train, test = train_test_split(series, test_size=test_size, shuffle=False)
    X_train, y_train = reshape(np.array(train))
    X_test, y_test = reshape(np.array(test))
    return X_train, y_train, X_test, y_test


def build_model(summary=True):
    model = Sequential()
    model.add(
        LSTM(
            input_shape=(window_size, num_categories),
            return_sequences=True,
            units=256))
    model.add(Dropout(_dropout))
    model.add(LSTM(256))
    model.add(Dropout(_dropout))
    model.add(Dense(num_categories, activation='sigmoid'))
    # model.add(Activation("tanh"))
    model.compile(
        loss='mean_squared_error',
        optimizer='adam',
        metrics=['accuracy'])
    if summary is True:
        model.summary()
    return model


raw_data = read_file()
num_data = to_numerical(raw_data)
enc_data = pd.DataFrame(to_categorical(num_data, num_classes=num_categories))
X_train, y_train, X_test, y_test = to_slidingwindow_series(enc_data, 3, 0.1)
model = build_model()
history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=1,
    verbose=1,
    validation_split=0.1)
yhat = model.predict(X_test)

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

np.argmax(yhat[0])
np.argmax(y_test[0])
yhat[0][28]
yhat[0][17]

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

# Save the history as a pickle object
with open('model_history.pkl', 'wb') as output:
    pickle.dump(history, output, pickle.HIGHEST_PROTOCOL)

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
