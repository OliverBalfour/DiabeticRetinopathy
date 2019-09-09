
import os, sys
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras.applications.densenet import preprocess_input
sys.path.append('models')
from model_utils import process_model_without_saving, process_model

for cnn in ['densenet121', 'mobilenet']:
	print('Loading ' + cnn)
	model = keras.models.load_model('models/h5/'+cnn+'-binary.h5')
	print('Processing ' + cnn)
	process_model(model, cnn+'-test-output', 'data/proc/binary/test/224/', 224, preprocess=preprocess_input, max_steps=1000, binary=True)
	# process_model_without_saving(model, cnn+'-test', 'data/proc/binary/test/224/', 224, preprocess=preprocess_input, max_steps=1000, binary=True)
