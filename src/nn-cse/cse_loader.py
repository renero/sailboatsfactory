from cse import CSE

cse = CSE().init()
cse.set_metadata('period', '1y')
cse.onehot_encode()
cse.to_slidingwindow_series(window_size=3, test_size=.1)

# Load or build a model
model = cse.load_model('./networks/model_20180827_100_0.75')
# cse.train(cse.build_model())

# Make predictions over the test set.
yhat = cse.predict()

# Save everything and finish.
cse.save_model(model,
               'model_{}'.format())
cse.plot_history()

# # evaluate loaded model on test data
# loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop',
#                      metrics=['accuracy'])
# score = loaded_model.evaluate(X, Y, verbose=0)
# print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

cse._metadata
cse.exists_output_name()
cse._input_file
