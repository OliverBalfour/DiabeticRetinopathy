
import numpy as np
import os, sys, pickle, contextlib, warnings
sys.path.append('./models')
sys.path.append('./models/stacked')
from model_utils import load_xy
from base_model import load_model
from sklearn.model_selection import train_test_split as tts

import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)

model_names = [fname[:-3] for fname in os.listdir('models/stacked/') if 'pycache' not in fname and 'base' not in fname]
model_modules = {name: __import__(name) for name in model_names}

vectors = {
	"densenet121": load_xy('densenet121'),
	"mobilenet": load_xy('mobilenet')
}

models = { "densenet121": [], "mobilenet": [] }

def train_test_split (X, Y, split=0.2, onehot=True):
	Xt, Xv, Yt, Yv = tts(X, Y, test_size=split, shuffle=True)
	if not onehot:
		Yt = np.argmax(Yt, axis=1)
		Yv = np.argmax(Yv, axis=1)
	return Xt, Xv, Yt, Yv

split_vectors = { "densenet121": [], "mobilenet": [] }

for cnn in vectors:
	print(f'Processing {cnn} stacked models')
	X, Y = vectors[cnn]
	Xt, Xv, Yt, Yv = train_test_split(X, Y, onehot=False)
	split_vectors[cnn] = (Xt, Xv, Yt, Yv)
	for model_name in model_modules:

		model = model_modules[model_name].Model()
		loaded = load_model(model.name, cnn)

		if loaded is not None:
			print(f'Loading {model_name.upper()}...', end=(' ' * (8-len(model_name))), flush=True)
			model = loaded
		else:
			print(f'Training {model_name.upper()}...', end=(' ' * (7-len(model_name))), flush=True)
			with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
				with warnings.catch_warnings():
					warnings.filterwarnings('ignore')
					model.train(Xt, Xv, Yt, Yv)
					model.save(cnn)

		models[cnn].append(model)
		print(str(round(model.acc*100, 2)) + '% acc')

# ANNs don't like being pickled :/
for key in models:
	for model in models[key]:
		if model.name == 'ANN':
			model.src = None

pickle.dump((models, split_vectors), open(f'models/pkl/all-models.pkl', 'wb'))

# save train/valid data used?
