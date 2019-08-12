
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD

# use batch norm and dropout
# write Model class
# implement train(X,Y) and predict(x)
# subclass for all methods

def train (X, Y, source, valid=None):
	model = Sequential()

#	model.add(Dense(256, activation='relu', input_shape=X.shape[1:]))
#	model.add(Dense(256, activation='relu'))
	model.add(Dense(2, activation='softmax', input_shape=X.shape[1:]))
	# WHY THE FUCK DO YOU HAVE 5 CLASSES FOR A BINARY PROBLEM THATS OBVIOUSLY WHY ITS NOT WORKING YOU BLOODY ^%*((*&$*%^$*&^^$(

	model.summary()

	model.compile(
		loss='categorical_crossentropy',
		optimizer='adam',
		metrics=['accuracy']
	)

	history = model.fit(
		X, Y,
		batch_size=50,
		epochs=20,
		steps_per_epoch=2000,
		verbose=1,
		shuffle=True
	)

	model.save(f'models/h5/ANN-{source}.h5')

	# choosing the mode 100% of the time gives:
	c = np.zeros(2)
	for row in Y:
		c[np.argmax(row)] += 1
	print(np.max(c)/np.sum(c))
	print(history.history['acc'][-1])
	return history.history
