
# SUPPORT VECTOR MACHINE

from base_model import BaseModel

import numpy as np
from sklearn import svm, metrics
from sklearn.model_selection import GridSearchCV

class Model (BaseModel):
	def __init__ (self):
		super().__init__('SVM')

	def train (self, Xt, Xv, Yt, Yv, verbose=False):
		params = {
			"kernel": ["linear"],
			"C": [1e-1],
			"gamma": [1e-4]
		}

		models = GridSearchCV(svm.SVC(verbose=verbose), params, verbose=verbose)
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
		return self.onehot_from_cat(self.src.predict(X))
