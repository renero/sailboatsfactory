from cpt import CPT


model = CPT()
train, test = model.load_files("training0.csv", "test0.csv", merge=True)
model.train(train)
predictions = model.predict(train, test, 3, 1)
print(predictions)

for t in zip(test, predictions):
    print(t[0],'->',t[1])
