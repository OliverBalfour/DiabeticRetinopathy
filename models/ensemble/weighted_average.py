
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

true_labels = outputs['true_labels']['test']

predictions = np.zeros((test_size,2))

def fn (acc):
	return max(0, acc - 0.5)

for cnn in outputs:
	for modelname in outputs[cnn]:
		model_preds = outputs[cnn][modelname]['test']
		predictions = predictions + model_preds * fn(outputs['acc'][cnn][modelname])

predictions = np.argmax(predictions, axis=1)

print(metrics.accuracy_score(true_labels, predictions))
