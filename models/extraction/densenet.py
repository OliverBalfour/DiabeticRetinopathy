
import os, sys
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras.applications.densenet import preprocess_input
sys.path.append('models')
from model_utils import process_model

"""
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
densenet121 (Model)          (None, 7, 7, 1024)        7037504
_________________________________________________________________
global_average_pooling2d (Gl (None, 1024)              0
_________________________________________________________________
batch_normalization (BatchNo (None, 1024)              4096
_________________________________________________________________
dropout (Dropout)            (None, 1024)              0
_________________________________________________________________
dense (Dense)                (None, 512)               524800
_________________________________________________________________
dropout_1 (Dropout)          (None, 512)               0
_________________________________________________________________
dense_1 (Dense)              (None, 256)               131328
_________________________________________________________________
dropout_2 (Dropout)          (None, 256)               0
_________________________________________________________________
dense_2 (Dense)              (None, 5)                 1285
=================================================================
Total params: 7,699,013
Trainable params: 7,613,317
Non-trainable params: 85,696
_________________________________________________________________
"""


# does .get_layer return a layer from densenet121?

base_model = keras.models.load_model('models/h5/densenet121-categorical.h5')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('dense_2').output)

print('Loaded model. Processing...')
process_model(model, 'densenet121', 'data/proc/binary/train/224/', 224, preprocess=preprocess_input, max_steps=100, binary=True)
# process_model(model, 'densenet121-test', 'data/proc/binary/test/224/', 224, preprocess=preprocess_input, max_steps=8000, binary=True)

# from models.model_utils import load_xy
# from models.stacked.ann import train as tr
# def train (name):
# 	X, Y = load_xy(name)
# 	tr(X, Y, name)
#
# train('densenet121')
