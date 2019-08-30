
import os, sys, pickle
sys.path.append('./models')
sys.path.append('./models/stacked')
from model_utils import load_xy

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
		print(f'Training {model_name}')
		model = model_modules[model_name].Model()
		model.train(X, Y)
		model.save(cnn)
		models[cnn].append(model)
		print(f'{model_name} acc: {str(model.acc*100)}')

print(models)
pickle.dump(models, open(f'models/pkl/all-models.pkl', 'wb'))
