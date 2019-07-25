
import os, sys
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
sys.path.append('models')
from model_utils import process_model

model = InceptionResNetV2(weights='imagenet', include_top=False, pooling='avg')
process_model(model, 'inception-resnet', 299, preprocess=preprocess_input, max_steps=5000)
