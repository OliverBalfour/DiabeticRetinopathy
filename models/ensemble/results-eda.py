
import pickle, sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
sys.path.append('./models')
from model_utils import load_xy

num_models = 10
data = pickle.load(open('models/pkl/all-stacked-outputs.pkl', 'rb'))
fig, axs = plt.subplots(2, num_models, figsize=(20, 6))
my_cmap = sns.light_palette("Navy", as_cmap=True)

print('Loaded data')

for y, cnn in enumerate(data.keys()):
	true_labels = data['true_labels']['valid']

	for x, model in enumerate(data[cnn].keys()):
		if x >= num_models: continue # NOTE: is this choosing the best 5 models?
		# TODO: note that ANNs weren't even in the top 5 models - this approach is better?
		# maybe make sure it's actually validation data?

		predictions = np.clip(np.round(data[cnn][model]['valid']), 0, 1)
		confmat = confusion_matrix(true_labels, predictions)

		ax = axs[y][x]
		sns.heatmap(confmat, ax=ax, annot=True, cmap=my_cmap, cbar=(x == num_models), fmt='d')

		ax.set_title(cnn + ' ' + model)
		if y == 1: ax.set_xlabel('Prediction')
		if x == 0: ax.set_ylabel('True Diagnosis')
		if y == 1: ax.xaxis.set_ticklabels(('benign', 'malignant'))
		else: ax.xaxis.set_ticklabels(('',''))
		if x == 0: ax.yaxis.set_ticklabels(('benign', 'malignant'))
		else: ax.yaxis.set_ticklabels(('',''))

		ax.set_ylim(bottom=2, top=0) # ensures the y axis is the right way around
		# TN FP
		# FN TP

print('Generated charts')

plt.subplots_adjust(
	top=0.95,
	bottom=0.07,
	left=0.04,
	right=0.98,
	hspace=0.2,
	wspace=0.1
)
plt.show()

print('Done')
