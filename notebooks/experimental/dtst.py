import numpy as np
from numpy import empty
import data
from sklearn.preprocessing import MinMaxScaler

interval = 1
a = np.array(([1,2,4,7,11,16,22,29,37,46],[1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0])).transpose()
print(a)
non_stationary = data.diff(a)
print(non_stationary)
params = dict()
params['lstm_predictions'] = 1
params['lstm_timesteps'] = 4
num_samples = non_stationary.shape[0]
num_features = non_stationary.shape[1]
num_predictions = params['lstm_predictions']
num_timesteps = params['lstm_timesteps']
num_frames = num_samples - (num_timesteps + num_predictions) + 1
# Update internal cache of parameters
params['num_samples'] = num_samples
params['num_features'] = num_features
params['num_frames'] = num_frames
params['x_scaler'] = MinMaxScaler(feature_range=(-1, 1))
params['y_scaler'] = MinMaxScaler(feature_range=(-1, 1))
params['num_testcases'] = 2

X = empty((num_frames, num_timesteps, num_features))
Y = empty((num_frames, num_predictions))
for i in range(num_samples - num_timesteps):
    X[i] = non_stationary[i:i + num_timesteps, ]
    Y[i] = non_stationary[i + num_timesteps:i + num_timesteps + num_predictions, 0]
Y
X_scaled = np.array([params['x_scaler'].fit_transform(X[i]) for i in range(X.shape[0])])
Y_scaled = params['y_scaler'].fit_transform(Y)
Y_scaled
X_train, Y_train, X_test, Y_test = data.split(X_scaled, Y_scaled, params['num_testcases'])
Y_test
y_unscaled = params['y_scaler'].inverse_transform(Y_test)
y_undiff = data.inverse_diff(y_unscaled, a[-(params['num_testcases']+1):, 0:1])[-params['num_testcases']:]
y_undiff

print(data.recover_Ytest(Y_test, a, params))







# hi
