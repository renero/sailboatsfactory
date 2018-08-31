from nn_cse.cs_nn import Csnn

cse_nn = Csnn().init('./nn_cse/params.yaml')
cse_nn.onehot_encode()
cse_nn.to_slidingwindow_series(window_size=3, test_size=.1)

# Load or build a model
model = cse_nn.load_model('./nn_cse/networks/model_20180827_100_0.75')
# cse_nn.train(cse_nn.build_model())

# Make predictions over the test set.
yhat = cse_nn.predict()

# Save everything and finish.
cse_nn.save_model()
cse_nn.plot_history()

# # evaluate loaded model on test data
# loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop',
#                      metrics=['accuracy'])
# score = loaded_model.evaluate(X, Y, verbose=0)
# print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
