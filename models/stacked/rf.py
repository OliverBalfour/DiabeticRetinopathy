
# RANDOM FOREST

from base_model import BaseModel

import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

class Model (BaseModel):
	def __init__ (self):
		super().__init__('RF')

	def train (self, X, Y, verbose=False):
		Xt, Xv, Yt, Yv = self.train_test_split(X, Y, onehot=False)

		params = {
			"max_depth": [None],
			"max_features": [200],
			"n_estimators": [30]
		}

		models = GridSearchCV(RandomForestClassifier(verbose=verbose), params, verbose=verbose)
		models.fit(Xt, Yt)
		if verbose: print('fit models')

		best = models.best_estimator_
		best.fit(Xt, Yt)
		if verbose: print('fit best model')

		pred = best.predict(Xv)
		acc = metrics.accuracy_score(Yv, pred)
		if verbose: print('Acc: ' + str(acc))

		self.src = best
		self.acc = acc

	def predict (self, X):
		return self.src.predict(X)
