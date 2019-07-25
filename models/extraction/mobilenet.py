
import os, sys
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input
sys.path.append('models')
from model_utils import process_model

model = MobileNet(weights='imagenet', include_top=False, pooling='avg')
process_model(model, 'mobilenet', 224, preprocess=preprocess_input, max_steps=8000)

# shape is :,7,7,1024 and we want global averages so :,1,1,1024 or equivalently :,1024
#np.moveaxis(nd,-1,1).mean(axis=-1).mean(axis=-1)
