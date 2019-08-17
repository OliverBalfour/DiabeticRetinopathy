
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization
from tensorflow.keras.optimizers import SGD
from sklearn.model_selection import train_test_split

# use batch norm and dropout
# write Model class
# implement train(X,Y) and predict(x)
# subclass for all methods

# no dropout seems to radically boost perf, is this correct?
def train (X, Y, source, valid=None):
	Xt, Xv, Yt, Yv = train_test_split(X, Y, test_size=0.1, shuffle=True)
	model = Sequential([
		Input(shape=X.shape[1:]),
#		Dense(256, activation='relu'),
#		BatchNormalization(),
#		Dropout(0.4),
		Dense(256, activation='relu'),
		BatchNormalization(),
		Dropout(0.4),
		Dense(2, activation='softmax')
	])

	model.summary()

	model.compile(
		loss='categorical_crossentropy',
		optimizer='adam',
		metrics=['accuracy']
	)

	history = model.fit(
		Xt, Yt,
		batch_size=50,
		epochs=5,
		steps_per_epoch=(X.shape[0] // 50),
		verbose=1,
		shuffle=True,
		validation_data=(Xv,Yv)
	)

	model.save(f'models/h5/ANN-{source}.h5')

	print(history.history['acc'][-1])
	if 'val_acc' in history.history.keys(): print(history.history['val_acc'][-1])
	return history.history
