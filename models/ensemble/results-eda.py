
import pickle, sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
sys.path.append('./models')
from model_utils import load_xy

data = pickle.load(open('models/pkl/stacked-outputs.pkl', 'rb'))

fig, axs = plt.subplots(2, 5, figsize=(16, 12))

print('Loaded data')

my_cmap = sns.light_palette("Navy", as_cmap=True)

for y, cnn in enumerate(data.keys()):
	true_labels = np.argmax(load_xy(cnn)[1], axis=1)

	for x, model in enumerate(data[cnn].keys()):
		if x >= 5: continue # NOTE: is this choosing the best 5 models?
		# TODO: note that ANNs weren't even in the top 5 models - this approach is better?


		predictions = np.clip(np.round(data[cnn][model]['train']), 0, 1)
		confmat = confusion_matrix(true_labels, predictions)

		ax = axs[y][x]
		sns.heatmap(confmat, ax=ax, annot=True, cmap=my_cmap, cbar=(x == 5), fmt='d')

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

plt.show()

print('Done')
