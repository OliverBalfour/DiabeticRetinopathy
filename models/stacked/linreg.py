
# LINEAR REGRESSION (LINREG)

import numpy as np
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, train_test_split
import pickle

def train (X, Y, source):
	Xt, Xv, Yt, Yv = train_test_split(X, Y, test_size=0.2, shuffle=True)
	Yt = np.argmax(Yt, axis=1)
	Yv = np.argmax(Yv, axis=1)

	model = LinearRegression(n_jobs=-1)
	model.fit(Xt, Yt)
	print('fit models')

	pred = np.round(model.predict(Xv))
	acc = metrics.accuracy_score(Yv, pred)
	print('Acc: ' + str(acc))

	pickle.dump(model, open(f'models/h5/LINREG-{source}.save', 'wb'))

	return acc
