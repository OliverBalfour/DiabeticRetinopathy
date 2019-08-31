
import os, sys, pickle, contextlib, warnings
sys.path.append('./models')
sys.path.append('./models/stacked')
from model_utils import load_xy
from base_model import load_model

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

for cnn in vectors:
	print(f'Processing {cnn} stacked models')
	X, Y = vectors[cnn]
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
					model.train(X, Y)
					model.save(cnn)

		models[cnn].append(model)
		print(str(round(model.acc*100, 2)) + '% acc')

# ANNs don't like being pickled :/
for key in models:
	for model in models[key]:
		if model.name == 'ANN':
			model.src = None

pickle.dump(models, open(f'models/pkl/all-models.pkl', 'wb'))
