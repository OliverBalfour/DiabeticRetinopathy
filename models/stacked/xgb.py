
# GRADIENT BOOSTING (XGBOOST)

import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import pickle

def train (X, Y, source):
	Xt, Xv, Yt, Yv = train_test_split(X, Y, test_size=0.2, shuffle=True)
	Yt = np.argmax(Yt, axis=1)
	Yv = np.argmax(Yv, axis=1)

	model = XGBClassifier(verbose=1)
	model.fit(Xt, Yt)
	print('fit model')

	pred = model.predict(Xv)
	acc = metrics.accuracy_score(Yv, pred)
	print('Acc: ' + str(acc))

	pickle.dump(model, open(f'models/h5/XGB-{source}.save', 'wb'))

	return acc
