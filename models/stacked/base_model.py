
import pickle
import numpy as np
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split as tts

def load_model (name, alias):
	try:
		model = pickle.load(open(f'models/pkl/{name}-{alias}.pkl', 'rb'))
		if name == 'ANN': model.src = keras.models.load_model(f'models/pkl/{name}-{alias}.h5')
		return model
	except Exception as e:
		print(e)
		return None

class BaseModel:
	def __init__ (self, name):
		self.name = name
		self.acc = 0 # should be validation accuracy
		self.src = None

	def train (self, X, Y, verbose=False):
		self.acc = 0
		self.src = None

	def predict (self, X):
		return None # must be categorical, not onehot

	def save (self, alias):
		pickle.dump(self, open(f'models/pkl/{self.name}-{alias}.pkl', 'wb'))

	def train_test_split (self, X, Y, split=0.1, onehot=True):
		Xt, Xv, Yt, Yv = tts(X, Y, test_size=split, shuffle=True)
		if not onehot:
			Yt = np.argmax(Yt, axis=1)
			Yv = np.argmax(Yv, axis=1)
		return Xt, Xv, Yt, Yv
