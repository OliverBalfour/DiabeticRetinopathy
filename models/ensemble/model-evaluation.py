

import os, sys, pickle, logging
import numpy as np
import tensorflow.keras as keras

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

# execute models on data to get predictions

outputs = { "densenet121": {}, "mobilenet": {}, 'true_labels': {} }

for cnn in models:
	Xt, Xv, Yt, Yv = split_vectors[cnn]
	Xte, Yte = load_xy(cnn + '-test')


	for model in models[cnn]:
		# get predictions for accuracy and confusion matrices
		outputs[cnn][model.name] = {
			"train": np.identity(2)[model.predict(Xt)],
			"valid": np.identity(2)[model.predict(Xv)],
			"test": np.identity(2)[model.predict(Xte)]
		}

outputs['true_labels'] = {
	"train": split_vectors['mobilenet'][2],
	"valid": split_vectors['mobilenet'][3],
	"test": load_xy(cnn + '-test')[1]
}

outputs['acc'] = { cnn: { model.name: model.acc for model in models[cnn] } for cnn in ['densenet121', 'mobilenet'] }

"""
outputs = {
	"mobilenet": { "ADA": { "train": (samples, features), "valid": ..., "test": ... }, ... }
	"densenet121": ...,
	"true_labels": { "train": (samples, 2), "valid": ..., "test": ... },
	"acc": { "mobilenet": { "ADA": 0.99, ... }, "densenet121": ... }
}
"""

pickle.dump(outputs, open('models/pkl/all-stacked-outputs.pkl', 'wb'))

# get predictions for accuracy, confusion matrices, local accuracy computation
# generate confusion matrices for each model, and naive ensembling methods
