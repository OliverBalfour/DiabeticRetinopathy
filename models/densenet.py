
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.applications.densenet import DenseNet121 as DenseNet
from model_utils import SequentialConstructor, evaluate, pretty_print_history

model = SequentialConstructor([
	DenseNet(weights='imagenet', include_top=False),
	GlobalAveragePooling2D(),
	BatchNormalization(),
	Dropout(0.5),
	Dense(512, activation='relu'),
	Dropout(0.5),
	Dense(256, activation='relu'),
	Dropout(0.5),
], output_shape=(5,))

hist = evaluate(model, model_name='densenet121-test')

pretty_print_history(hist)
