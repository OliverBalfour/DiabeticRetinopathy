
import os, sys, pickle
import numpy as np
from sklearn import metrics

outputs = pickle.load(open('models/pkl/all-stacked-outputs.pkl', 'rb'))

test_size = 2000
num_cnns = 2
num_models = len(outputs['mobilenet'].keys())
true_labels = outputs['true_labels']['mobilenet']['test']

# returns ndarray (models, samples, classes) and list of model accuracies
def get_prediction_tensor ():
	tensor = np.zeros((num_models*num_cnns, test_size, 2))
	models = []
	for cnn in  ['densenet121', 'mobilenet']:
		for modelname in outputs[cnn]:
			tensor[len(models)] = outputs[cnn][modelname]['test']
			models.append(outputs['acc'][cnn][modelname])
			# models.append((cnn, modelname, outputs['acc'][cnn][modelname]))
	return tensor, np.array(models)

tensor, models = get_prediction_tensor()

# ensemble with a specific averaging function which is passed an ndarray (models, classes) and model accuracies for each sample and prints accuracy #returns (classes,)
def ensemble_predictions (tensor, models, fn):
	predictions = [fn(tensor[:,k,:], models) for k in range(tensor.shape[1])]
	predictions = np.argmax(predictions, axis=1)
	print(metrics.accuracy_score(np.argmax(true_labels, axis=1), predictions))

ensemble_predictions(tensor, models, lambda mat, models: np.average(mat.T, axis=1).T)
ensemble_predictions(tensor, models, lambda mat, models: np.sum(mat.T, axis=1).T)
ensemble_predictions(tensor, models, lambda mat, models: np.sum(np.multiply(mat.T, np.max(models-0.6)), axis=1).T)

# for some reason the acc weighted code gives identical results... fix?
