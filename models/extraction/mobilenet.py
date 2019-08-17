
import os, sys
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras.applications.densenet import preprocess_input
sys.path.append('models')
from model_utils import process_model


base_model = keras.models.load_model('models/h5/mobilenet-binary.h5')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('batch_normalization').output)

print('Loaded model. Processing...')
process_model(model, 'mobilenet', 'data/proc/binary/train/224/', 224, preprocess=preprocess_input, max_steps=8000, binary=True)
process_model(model, 'mobilenet-test', 'data/proc/binary/test/224/', 224, preprocess=preprocess_input, max_steps=1000, binary=True)
