
import os, sys, pickle, logging
import numpy as np
import tensorflow.keras as keras

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)

sys.path.append('./models')
sys.path.append('./models/stacked')

models, _ = pickle.load(open('models/pkl/all-models.pkl', 'rb'))

print('Loaded all stacked models')

num_models = 6

chosen_models = { "densenet121": [], "mobilenet": [] }

for cnn in models:
	accuracies = sorted([(model.name, model.acc) for model in models[cnn]], key=lambda x: x[1], reverse=True)
	print(cnn)
	print('\n'.join([' ' + name + ' '*(8-len(name)) + str(round(acc*100,2)) + '%' for name, acc in accuracies]))
	names = [name for name, acc in accuracies[:num_models]]
	print('Using the following '+str(num_models)+' models: ' + ', '.join(names))
	best_models = sorted([model for model in models[cnn] if model.name in names], key=lambda m: m.acc, reverse=True)
	chosen_models[cnn] = best_models

pickle.dump(chosen_models, open(f'models/pkl/best-models.pkl', 'wb'))
