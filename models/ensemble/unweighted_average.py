
import os, sys, pickle
import numpy as np
from sklearn import metrics

sys.path.append('./models')
sys.path.append('./models/stacked')

from model_utils import load_xy

outputs = pickle.load(open('models/pkl/all-stacked-outputs.pkl', 'rb'))

test_size = 2000
num_cnns = 2
num_models = len(outputs['mobilenet'].keys())

true_labels = np.argmax(load_xy('densenet121-test')[1], axis=1)

predictions = np.zeros((test_size,2))

for cnn in outputs:
	for modelname in outputs[cnn]:
		model_cat = np.asarray(outputs[cnn][modelname]['test'], dtype='int') # remove asarray and onehot
		model_onehot = np.identity(2)[model_cat]
		predictions = predictions + model_onehot / (num_cnns * num_models)

predictions = np.argmax(predictions, axis=1)

print(metrics.accuracy_score(true_labels, predictions))
