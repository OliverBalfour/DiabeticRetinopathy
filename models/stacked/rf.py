
# RANDOM FOREST

import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
import pickle

def train (X, Y, source):
	Xt, Xv, Yt, Yv = train_test_split(X, Y, test_size=0.2, shuffle=True)
	Yt = np.argmax(Yt, axis=1)
	Yv = np.argmax(Yv, axis=1)

	params = {
		"max_depth": [None],
		"max_features": [200],
		"n_estimators": [30]
	}

	models = GridSearchCV(RandomForestClassifier(verbose=1), params, verbose=1)
	models.fit(Xt, Yt)
	print('fit models')

	best = models.best_estimator_
	best.fit(Xt, Yt)
	print('fit best model')

	pred = best.predict(Xv)
	acc = metrics.accuracy_score(Yv, pred)
	print('Acc: ' + str(acc))

	pickle.dump(best, open(f'models/h5/RF-{source}.save', 'wb'))

	return acc
