
# ARTIFICIAL NEURAL NETWORK

from base_model import BaseModel
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

class Model (BaseModel):
	def __init__ (self):
		super().__init__('ANN')

	def train (self, X, Y, verbose=False):
		Xt, Xv, Yt, Yv = self.train_test_split(X, Y)
		model = Sequential([
			Dense(256, input_shape=X.shape[1:], activation='relu'),
			BatchNormalization(),
			Dropout(0.4),
			Dense(2, activation='softmax')
		])

		if verbose: model.summary()

		model.compile(
			loss='categorical_crossentropy',
			optimizer='adam',
			metrics=['accuracy']
		)

		history = model.fit(
			Xt, Yt,
			batch_size=50,
			epochs=2,
			steps_per_epoch=(X.shape[0] // 50),
			verbose=1,
			shuffle=True,
			validation_data=(Xv,Yv)
		)

		self.src = model
		self.acc = history.history['val_acc'][-1]

	def predict (self, X):
		return np.argmax(self.src.predict(X), axis=1)

	def save (self, alias):
		# save actual model separately in h5 file
		self.src.save(f'models/pkl/{self.name}-{alias}.h5')
		model = self.src
		self.src = None
		super().save(alias)
		self.src = model
