
# ADAPTIVE BOOSTING (ADABOOST)

from base_model import BaseModel

import numpy as np
from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV, train_test_split

class Model (BaseModel):
	def __init__ (self):
		super().__init__('ADA')

	def train (self, X, Y, verbose=False):
		Xt, Xv, Yt, Yv = self.train_test_split(X, Y, onehot=False)

		models = GridSearchCV(AdaBoostClassifier(), { "n_estimators": [30] }, verbose=verbose)
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
