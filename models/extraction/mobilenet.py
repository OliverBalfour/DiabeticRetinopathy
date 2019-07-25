
import os, sys
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
sys.path.append('models')
from model_utils import process_model

model = MobileNet(weights='imagenet', include_top=False, pooling='avg')
process_model(model, 'mobilenet', 224, preprocess=preprocess_input, max_steps=8000)
