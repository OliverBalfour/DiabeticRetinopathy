
# SUPPORT VECTOR MACHINE

import numpy as np
from sklearn import svm, metrics
from sklearn.model_selection import GridSearchCV, train_test_split
import pickle

def train (X, Y, source):
	Xt, Xv, Yt, Yv = train_test_split(X, Y, test_size=0.7, shuffle=True)
	Yt = np.argmax(Yt, axis=1)
	Yv = np.argmax(Yv, axis=1)

	params = {
		"kernel": ["linear"],
		"C": [1e-1],
		"gamma": [1e-4]
	}

	models = GridSearchCV(svm.SVC(verbose=1), params, verbose=1)
	models.fit(Xt, Yt)
	print('fit models')

	best = models.best_estimator_
	best.fit(Xt, Yt)
	print('fit best model')

	pred = best.predict(Xv)
	acc = metrics.accuracy_score(Yv, pred)
	print('Acc: ' + str(acc))

	pickle.dump(best, open(f'models/h5/SVM-{source}.save', 'wb'))

	return acc
