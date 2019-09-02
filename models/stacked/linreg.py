
# LINEAR REGRESSION (LINREG)

from base_model import BaseModel

import numpy as np
from sklearn import metrics
from sklearn.linear_model import LinearRegression

class Model (BaseModel):
	def __init__ (self):
		super().__init__('LINREG')

	def train (self, Xt, Xv, Yt, Yv, verbose=False):
		model = LinearRegression()
		model.fit(Xt, Yt)
		if verbose: print('fit models')

		pred = np.round(model.predict(Xv))
		acc = metrics.accuracy_score(Yv, pred)
		if verbose: print('Acc: ' + str(acc))

		self.src = model
		self.acc = acc

	def predict (self, X):
		return self.onehot_from_cat(self.src.predict(X))
