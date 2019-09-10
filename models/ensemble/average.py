
import os, sys, pickle
import numpy as np
from sklearn import metrics
from lehmer_mean import optimise_lehmer_mean, lehmer_mean
from local_accuracy import compute_local_accuracy, get_nearest_neighbours_densenet
sys.path.append('./models')
from model_utils import load_xy
import matplotlib.pyplot as plt

# NOTE: this code is super messy and needs refactoring

outputs = pickle.load(open('models/pkl/all-stacked-outputs.pkl', 'rb'))

X_densenet, _ = load_xy('densenet121')

test_size = 2000
num_cnns = 2
num_models = len(outputs['mobilenet'].keys())
true_labels = outputs['true_labels']['mobilenet']['test']

# returns ndarray (models, samples, classes) and list of model accuracies
def get_prediction_tensor ():
	tensor = np.zeros((num_models*num_cnns, test_size, 2))
	models = []
	modelnames = []
	for cnn in ['densenet121', 'mobilenet']:
		for modelname in outputs[cnn]:
			tensor[len(models)] = outputs[cnn][modelname]['test']
			models.append(outputs['acc'][cnn][modelname])
			modelnames.append(modelname)
			# models.append((cnn, modelname, outputs['acc'][cnn][modelname]))
	return tensor, np.array(models), modelnames

# tensor is tensor of predictions, models is list of model accuracies
# modelnames is a list of stacked model names, densenet first then mobilenet
tensor, models, modelnames = get_prediction_tensor()

# ensemble with a specific averaging function which is passed an ndarray (models, classes) and model accuracies and index for each sample and returns accuracy
def ensemble_predictions (tensor, models, fn):
	predictions = [fn(tensor[:,k,:], models, k) for k in range(tensor.shape[1])]
	predictions = np.argmax(predictions, axis=1)
	tn, fp, fn, tp = metrics.confusion_matrix(np.argmax(true_labels, axis=1), predictions).ravel()
	return {
		"accuracy": round(metrics.accuracy_score(np.argmax(true_labels, axis=1), predictions) * 100, 2),
		"sensitivity": round(tp/(tp+fn)*100, 2),
		"specificity": round(tn/(tn+fp)*100, 2)
	}

def en (title, fn):
	data = ensemble_predictions(tensor, models, fn)
	spaces = ' ' * (50 - len(title))
	print(f'{title}: {spaces} {data["accuracy"]}%      {data["sensitivity"]}%      {data["specificity"]}%')
print('Model' + ' ' * 47 + 'accuracy sensitivity specificity')

epsilon = 0.5

# returns optimised base p for Lehmer mean using a specific cnn and dataset
def get_p (cnn, dataset):
	return optimise_lehmer_mean(
		np.array([outputs[cnn][modelname][dataset] for modelname in sorted(outputs[cnn].keys())]).swapaxes(0,1) + epsilon,
		np.identity(2)[outputs['true_labels'][cnn][dataset]] + epsilon,
		w=np.clip(np.array([outputs['acc'][cnn][modelname] for modelname in sorted(outputs[cnn].keys())]) - 0.6, 0, 1)**3
	)

# takes a matrix (models, classes) and vec (models) and returns (classes,) using lehmer mean across columns
def lehmer_mat (mat, accuracies, index, weights=None):
	class_vecs = [c for c in mat.T]
	default_weights = np.clip(accuracies-0.6,0.01,1)**3
	weights = weights if weights is not None else default_weights #np.repeat(default_weights[np.newaxis, ...], test_size, axis=0)
	densenet_mat = np.transpose([lehmer_mean(c + epsilon, p_densenet, w=weights) for c in mat.T])
	mobilenet_mat = np.transpose([lehmer_mean(c + epsilon, p_mobilenet, w=weights) for c in mat.T])
	accs = [0.708, 0.675] # from results-eda.py
	weights = np.array(accs) / np.sum(accs)
	return densenet_mat * weights[0] + mobilenet_mat * weights[1]

# takes a matrix (models, classes) and vec (models) and returns (classes,) using lehmer mean across columns
def lehmer_mat_custom (mat, accuracies, p):
	class_vecs = [c for c in mat.T]
	weights = np.clip(accuracies-0.6,0.01,1)
	return np.transpose([lehmer_mean(c + epsilon, p, w=weights) for c in mat.T])

en('Best individual model', lambda mat, models, i: mat[np.argmax(models)])
en('Simple unweighted average', lambda mat, models, i: np.average(mat.T, axis=1).T)
en('Average weighted by accuracy', lambda mat, models, i: np.average(np.multiply(mat.T, np.clip(models-0.6,0,1)), axis=1).T)
for k in range(5):
	p = 0.6 + k/5
	en('L_p weighted by accuracy, p='+str(p), lambda mat, models, i: lehmer_mat_custom(mat, models, p))
for p in [-10, 0.01, 0.5, 10, -1000, 1000, -2.6, 4.8]:
	en('L_p weighted by accuracy, p='+str(p), lambda mat, models, i: lehmer_mat_custom(mat, models, p))

# takes ages to compute, and the output is the same every time, hence the float literals
p_densenet = 1.1756283000118866 #get_p('densenet121', 'valid')
# print(p_densenet)
p_mobilenet = 0.7142255422715837 #get_p('mobilenet', 'valid')
# print(p_mobilenet)

en('Optimised Lehmer mean weighted by accuracy', lehmer_mat)

def generate_local_accuracies ():
	all_la = []
	for k in range(tensor.shape[1]):
		mat = tensor[:,k,:]
		accuracies = models
		i = k

		x = X_densenet[i]
		indices, inverse_squares = get_nearest_neighbours_densenet(X_densenet[i], i)
		la = [] # local accuracies
		for i, modelname in enumerate(modelnames):
			cnn = ['densenet121', 'mobilenet'][int(i > len(modelnames)//2)]
			Y_hat = outputs[cnn][modelname]['all-train']
			la.append(compute_local_accuracy(Y_hat, X_densenet[i], indices, inverse_squares))
		weights = np.clip(np.array(la)-0.5,0.01,1)**3
		all_la.append(weights)
	return np.array(all_la)

la = generate_local_accuracies()

def local_acc_weighted_lehmer_mean (mat, accuracies, i):
	return lehmer_mat(mat, accuracies, None, weights=la[i])

def local_acc_weighted_mean (mat, accuracies, i):
	return np.average(np.multiply(mat.T, la[i]*1/4 + accuracies *3/4), axis=1).T

en('Optimised Lehmer mean weighted by local accuracy', local_acc_weighted_lehmer_mean)
en('Average weighted by local accuracy', local_acc_weighted_mean)

print('Best CNN (densenet121):                             70.8%')

def graph_lehmer_base ():
	ps_pos = np.logspace(-1.0, 2.0, num=50)
	ps = np.concatenate((-ps_pos[::-1], ps_pos))
	all_data = {
		"accuracy": [],
		"sensitivity": [],
		"specificity": []
	}
	for p in ps:
		data = ensemble_predictions(tensor, models, lambda mat, models, i: lehmer_mat_custom(mat, models, p))
		all_data['accuracy'].append(data['accuracy'])
		all_data['sensitivity'].append(data['sensitivity'])
		all_data['specificity'].append(data['specificity'])

	plt.xscale('symlog')
	plt.yscale('linear')
	plt.ylim(top=100, bottom=50)
	for key in all_data:
		plt.plot(ps, all_data[key], label=key[0].upper()+key[1:])
	plt.legend()
	plt.title('Performance Metrics versus Lehmer Mean Base')
	plt.ylabel('Percentage (linear scale)')
	plt.xlabel('Base (p) for Lehmer mean (log scale)')
	plt.show()
