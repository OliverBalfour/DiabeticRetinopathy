
import os, sys
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras.applications.densenet import preprocess_input
sys.path.append('models')
from model_utils import process_model


#NOTE : THIS SUMMARY IS FOR CATEGORICAL
"""
densenet121 (Model)          (None, 7, 7, 1024)        7037504
global_average_pooling2d (Gl (None, 1024)              0
batch_normalization (BatchNo (None, 1024)              4096
dropout (Dropout)            (None, 1024)              0
dense (Dense)                (None, 512)               524800
dropout_1 (Dropout)          (None, 512)               0
dense_1 (Dense)              (None, 256)               131328
dropout_2 (Dropout)          (None, 256)               0
dense_2 (Dense)              (None, 5)                 1285
"""

base_model = keras.models.load_model('models/h5/densenet121-binary.h5')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('batch_normalization').output)

print('Loaded model. Processing...')
# process_model(model, 'densenet121', 'data/proc/binary/train/224/', 224, preprocess=preprocess_input, max_steps=8000, binary=True)
process_model(model, 'densenet121-test', 'data/proc/binary/test/224/', 224, preprocess=preprocess_input, max_steps=1000, binary=True)
