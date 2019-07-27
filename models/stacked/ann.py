
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

	model.add(Dense(X.shape[1], activation='relu', input_shape=X.shape[1:]))

	for layer in [512,512]:
		model.add(Dense(layer, activation='relu'))

	model.add(Dense(5, activation='softmax'))

	model.summary()

	model.compile(
		loss='categorical_crossentropy',
		optimizer=SGD(),
		metrics=['accuracy']
	)

	history = model.fit(
		np.clip(100*X,0,1), Y,
		batch_size=X.shape[0]//100,
		epochs=20,
		verbose=1,
		validation_data=valid,
		shuffle=True
	)

	model.save(f'models/h5/ANN-{source}.h5')

	# choosing the mode 100% of the time gives:
	c = np.zeros(5)
	for row in Y:
		c[np.argmax(row)] += 1
	print(np.max(c)/np.sum(c))
	print(history.history['acc'][-1])
	return history.history

# from models.model_utils import load_xy
# from models.stacked.ann import train as tr
# def train (name):
# 	X, Y = load_xy(name)
# 	tr(X, Y, name)
#
# train('mobilenet')
