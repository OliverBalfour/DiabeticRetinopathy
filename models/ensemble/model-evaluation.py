

import os, sys, pickle, logging
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.models import Model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)

sys.path.append('./models')
sys.path.append('./models/stacked')

from model_utils import load_xy

models, split_vectors = pickle.load(open('models/pkl/all-models.pkl', 'rb'))

# ANNs don't like being pickled :/
for cnn in models:
	for model in models[cnn]:
		if model.name == 'ANN':
			model.src = keras.models.load_model(f'models/pkl/ANN-{cnn}.h5')

print('Loaded all stacked models')

# execute models on data to get predictions

# TODO: eval outputs of CNNs themselves (compute stacked ANN on CNN vectors - requires loading CNN h5's)

outputs = { "densenet121": {}, "mobilenet": {}, 'true_labels': {} }

for cnn in models:
	Xt, Xv, Yt, Yv = split_vectors[cnn]
	Xte, Yte = load_xy(cnn + '-test')


	for model in models[cnn]:
		# get predictions for accuracy and confusion matrices
		outputs[cnn][model.name] = {
			"train": model.predict(Xt),
			"valid": model.predict(Xv),
			"test": model.predict(Xte)
		}

print('Evaluated all stacked models')

# for cnn in models:
# 	Xt, Xv, Yt, Yv = split_vectors[cnn]
# 	Xte, Yte = load_xy(cnn + '-test')
#
# 	base_model = keras.models.load_model(f'models/h5/{cnn}-binary.h5')
# 	model = Model(inputs=base_model.get_layer('batch_normalization').input, outputs=base_model.output)
#
# 	outputs[cnn]['cnn'] = {
# 		"train": model.predict(Xt),
# 		"valid": model.predict(Xv),
# 		"test": model.predict(Xte)
# 	}
# print('Evaluated both CNNs')

# these are categorical for some reason
# also remember that different valid datasets are used for the two CNNs
outputs['true_labels'] = {
	"densenet121": {
		"train": split_vectors['densenet121'][2],
		"valid": split_vectors['densenet121'][3],
		"test": load_xy('densenet121-test')[1]
	}, "mobilenet": {
		"train": split_vectors['mobilenet'][2],
		"valid": split_vectors['mobilenet'][3],
		"test": load_xy('mobilenet-test')[1]
	}
}

outputs['acc'] = { cnn: { model.name: model.acc for model in models[cnn] } for cnn in ['densenet121', 'mobilenet'] }

"""
outputs = {
	"mobilenet": { "ADA": { "train": (samples, features), "valid": ..., "test": ... }, ... }
	"densenet121": ...,
	"true_labels": { "densenet121": { "train": (samples, 2), "valid": ..., "test": ... }, ...},
	"acc": { "mobilenet": { "ADA": 0.99, ... }, "densenet121": ... }
}
"""

pickle.dump(outputs, open('models/pkl/all-stacked-outputs.pkl', 'wb'))

# get predictions for accuracy, confusion matrices, local accuracy computation
# generate confusion matrices for each model, and naive ensembling methods
