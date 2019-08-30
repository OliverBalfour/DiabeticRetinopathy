
# GRADIENT BOOSTING (XGBOOST)

from base_model import BaseModel

import numpy as np
from sklearn import metrics
from xgboost import XGBClassifier

class Model (BaseModel):
	def __init__ (self):
		super().__init__('XGB')

	def train (self, X, Y, verbose=False):
		Xt, Xv, Yt, Yv = self.train_test_split(X, Y, onehot=False)

		model = XGBClassifier(verbose=verbose)
		model.fit(Xt, Yt)
		if verbose: print('fit model')

		pred = model.predict(Xv)
		acc = metrics.accuracy_score(Yv, pred)
		if verbose: print('Acc: ' + str(acc))

		self.src = model
		self.acc = acc

	def predict (self, X):
		return self.src.predict(X)
