
import pickle, sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
sys.path.append('./models')
from model_utils import load_xy
from math import ceil

data = pickle.load(open('models/pkl/all-stacked-outputs.pkl', 'rb'))
my_cmap = sns.light_palette("Navy", as_cmap=True)

print('Loaded data')

model_names = {
	"ADA": "AdaBoost",
	"ANN": "Artificial Neural Network",
	"KM": "K-Means",
	"KNN": "K-Nearest Neighbours",
	"LINREG": "Linear Regression",
	"LOGREG": "Logistic Regression",
	"NB": "Naive Bayes",
	"RF": "Random Forest",
	"SVM": "Support Vector Machine",
	"XGB": "Gradient Boosting",
	"densenet121": "DenseNet",
	"mobilenet": "MobileNet",
	"A": "Base CNN"
}

cnns = ['densenet121', 'mobilenet']

num_models = 11
row_w = ceil(num_models/2)
fig = plt.figure(figsize=(20, 12))

for cnn in cnns:
	# 'A' for alphabetical ordering precedence
	data[cnn]['A'] = { 'test': load_xy(cnn+'-test-output')[0] }

for y, cnn in enumerate(cnns): # can't use data.keys() as this includes 'true_labels' and 'acc'
	true_labels = data['true_labels'][cnn]['test']

	for x, model in enumerate(sorted(data[cnn].keys())):
		if x >= num_models: continue

		predictions = data[cnn][model]['test']
		predictions = np.clip(np.round(predictions), 0, 1)
		truth = np.argmax(true_labels, axis=1) if model != 'A' else np.argmax(load_xy(cnn+'-test-output')[1], axis=1)
		confmat = confusion_matrix(truth, np.argmax(predictions, axis=1))

		acc = round(accuracy_score(truth, np.argmax(predictions, axis=1))*100,2)
		tn, fp, fn, tp = confmat.ravel()
		sensitivity = round(tp/(tp+fn)*100, 2)
		specificity = round(tn/(tn+fp)*100, 2)
		print(f'{cnn}-{model} test acc {str(acc)}% sensitivity {str(sensitivity)}% specificity {str(specificity)}%')

		fx = x % row_w
		fy = y*2 + x // row_w
		ax = fig.add_subplot(4, row_w, fy * row_w + fx + 1)
		sns.heatmap(confmat, ax=ax, annot=True, cmap=my_cmap, cbar=(x == num_models), fmt='d')

		ax.set_title(f'{model_names[cnn]} {model_names[model]} ({str(acc)}%)', fontsize=10)

		if x // row_w == 1 and y == 1: ax.set_xlabel('Prediction')
		if x // row_w == 1 and y == 1: ax.xaxis.set_ticklabels(('benign', 'malignant'), fontsize=8)
		else: ax.xaxis.set_ticklabels(('',''))

		if x % row_w == 0: ax.set_ylabel('True Diagnosis')
		if x % row_w == 0: ax.yaxis.set_ticklabels(('benign', 'malignant'), fontsize=8)
		else: ax.yaxis.set_ticklabels(('',''))


		ax.set_ylim(bottom=2, top=0) # ensures the y axis is the right way around
		# TN FP
		# FN TP

print('Generated charts')

# plt.subplots_adjust(
# 	top=0.95,
# 	bottom=0.07,
# 	left=0.04,
# 	right=0.98,
# 	hspace=0.2,
# 	wspace=0.1
# )
plt.tight_layout()
plt.subplots_adjust(hspace=0.3, wspace=0.2)
plt.show()

print('Done')
