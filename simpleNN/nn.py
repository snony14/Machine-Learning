from keras.models import Sequential
from keras.layers import Dense
import numpy as np

numpy.random.seed(7)

dataset = np.loadtxt("pima-indians-diabetes.data.csv", delimiter=',')
# Split into inpit (X) and output (Y) variables

X = dataset[:,0:8]
Y = dataset[:,8]

# Create model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#compile models
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#Fit the model
model.fit(X, Y, epochs=150, batch_size=10)

#evaluate model
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# calculate predictions
predictions = model.predict(X)
# round predictions
rounded = [round(x[0]) for x in predictions]
print(rounded)
