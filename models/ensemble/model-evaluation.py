

import os, sys, pickle, logging
import numpy as np
import tensorflow.keras as keras

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)

sys.path.append('./models')
sys.path.append('./models/stacked')

from model_utils import load_xy

models = pickle.load(open('models/pkl/best-models.pkl', 'rb'))

# ANNs don't like being pickled :/
for cnn in models:
	for model in models[cnn]:
		if model.name == 'ANN':
			model.src = keras.models.load_model(f'models/pkl/ANN-{cnn}.h5')

# execute models on data to get predictions

outputs = { "densenet121": {}, "mobilenet": {} }

for cnn in models:
	Xt, Yt = load_xy(cnn)
	Xv, Yv = load_xy(cnn + '-test')

	for model in models[cnn]:
		# get predictions for accuracy and confusion matrices
		outputs[cnn][model.name] = {
			"train": model.predict(Xt),
			"test": model.predict(Xv)
		}

pickle.dump(outputs, open(f'models/pkl/stacked-outputs.pkl', 'wb'))

# get predictions for accuracy, confusion matrices, local accuracy computation
# generate confusion matrices for each model, and naive ensembling methods
