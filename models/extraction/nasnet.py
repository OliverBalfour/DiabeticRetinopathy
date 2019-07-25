
import os, sys
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.applications.nasnet import NASNetMobile, preprocess_input
sys.path.append('models')
from model_utils import process_model

model = NASNetMobile(weights='imagenet', include_top=False, pooling='avg')
process_model(model, 'nasnet', 224, preprocess=preprocess_input, max_steps=5000)
