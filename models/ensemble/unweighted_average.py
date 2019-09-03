
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

true_labels = outputs['true_labels']['mobilenet']['test']

predictions = np.zeros((test_size,2))

avg = input('(w)eighted/(u)nweighted: ')

def weighted (acc):
	return max(0, acc - 0.6)
def unweighted (acc):
	return 1  / (num_cnns * num_models)

for cnn in ['densenet121', 'mobilenet']:
	for modelname in outputs[cnn]:
		model_preds = outputs[cnn][modelname]['test']
		acc = outputs['acc'][cnn][modelname]
		predictions = predictions + model_preds * (weighted(acc) if avg == 'w' else unweighted(acc))

predictions = np.argmax(predictions, axis=1)

print(metrics.accuracy_score(np.argmax(true_labels, axis=1), predictions))
