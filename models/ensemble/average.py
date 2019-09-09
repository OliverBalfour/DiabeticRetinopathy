
import os, sys, pickle
import numpy as np
from sklearn import metrics
from lehmer_mean import optimise_lehmer_mean, lehmer_mean

outputs = pickle.load(open('models/pkl/all-stacked-outputs.pkl', 'rb'))

test_size = 2000
num_cnns = 2
num_models = len(outputs['mobilenet'].keys())
true_labels = outputs['true_labels']['mobilenet']['test']

# returns ndarray (models, samples, classes) and list of model accuracies
def get_prediction_tensor ():
	tensor = np.zeros((num_models*num_cnns, test_size, 2))
	models = []
	for cnn in ['densenet121', 'mobilenet']:
		for modelname in outputs[cnn]:
			tensor[len(models)] = outputs[cnn][modelname]['test']
			models.append(outputs['acc'][cnn][modelname])
			# models.append((cnn, modelname, outputs['acc'][cnn][modelname]))
	return tensor, np.array(models)

tensor, models = get_prediction_tensor()

# ensemble with a specific averaging function which is passed an ndarray (models, classes) and model accuracies for each sample and returns accuracy
def ensemble_predictions (tensor, models, fn):
	predictions = [fn(tensor[:,k,:], models) for k in range(tensor.shape[1])]
	predictions = np.argmax(predictions, axis=1)
	return metrics.accuracy_score(np.argmax(true_labels, axis=1), predictions)

def en (title, fn):
	print(title + ': ' + ' ' * (50 - len(title)) + str(round(ensemble_predictions(tensor, models, fn) * 100, 2)) + '%')

epsilon = 0.5

# returns optimised base p for Lehmer mean using a specific cnn and dataset
def get_p (cnn, dataset):
	return optimise_lehmer_mean(
		np.array([outputs[cnn][modelname][dataset] for modelname in sorted(outputs[cnn].keys())]).swapaxes(0,1) + epsilon,
		np.identity(2)[outputs['true_labels'][cnn][dataset]] + epsilon,
		w=np.clip(np.array([outputs['acc'][cnn][modelname] for modelname in sorted(outputs[cnn].keys())]) - 0.6, 0, 1)**3
	)

# takes a matrix (models, classes) and vec (models) and returns (classes,) using lehmer mean across columns
def lehmer_mat (mat, accuracies):
	class_vecs = [c for c in mat.T]
	weights = np.clip(accuracies-0.6,0.01,1)**3
	densenet_mat = np.transpose([lehmer_mean(c + epsilon, p_densenet, w=weights) for c in mat.T])
	mobilenet_mat = np.transpose([lehmer_mean(c + epsilon, p_mobilenet, w=weights) for c in mat.T])
	return (densenet_mat + mobilenet_mat) / 2

# takes a matrix (models, classes) and vec (models) and returns (classes,) using lehmer mean across columns
def lehmer_mat_custom (mat, accuracies, p):
	class_vecs = [c for c in mat.T]
	weights = np.clip(accuracies-0.6,0.01,1)
	return np.transpose([lehmer_mean(c + epsilon, p, w=weights) for c in mat.T])

en('Best individual model', lambda mat, models: mat[np.argmax(models)])
en('Simple unweighted average', lambda mat, models: np.average(mat.T, axis=1).T)
en('Average weighted by accuracy', lambda mat, models: np.average(np.multiply(mat.T, np.clip(models-0.6,0,1)), axis=1).T)
for k in range(5):
	p = 0.6 + k/5
	en('L_p weighted by accuracy, p='+str(p), lambda mat, models: lehmer_mat_custom(mat, models, p))
for p in [-10, 0.01, 0.5, 10, -1000, 1000]:
	en('L_p weighted by accuracy, p='+str(p), lambda mat, models: lehmer_mat_custom(mat, models, p))

p_densenet = get_p('densenet121', 'valid')
print(p_densenet)
p_mobilenet = get_p('mobilenet', 'valid')
print(p_mobilenet)
en('Optimised Lehmer mean weighted by accuracy', lehmer_mat)
