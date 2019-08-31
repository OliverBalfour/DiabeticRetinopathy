
# K MEANS (KM)

from base_model import BaseModel

import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans

class Model (BaseModel):
	def __init__ (self):
		super().__init__('KM')

	def train (self, X, Y, verbose=False):
		Xt, Xv, Yt, Yv = self.train_test_split(X, Y, split=0.3, onehot=False)

		model = KMeans(n_clusters=2, n_jobs=-1)
		model.fit(Xt, Yt)
		if verbose: print('fit model')

		pred = model.predict(Xv)
		acc = metrics.accuracy_score(Yv, pred)
		if verbose: print('Acc: ' + str(acc))

		self.src = model
		self.acc = acc

	def predict (self, X):
		return self.src.predict(X)
