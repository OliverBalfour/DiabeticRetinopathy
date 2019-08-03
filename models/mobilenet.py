
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2 as MobileNet
from model_utils import SequentialConstructor, evaluate, pretty_print_history

model = SequentialConstructor([
	MobileNet(weights='imagenet', include_top=False, input_shape=image_shape),
	GlobalAveragePooling2D(),
	BatchNormalization(),
	Dropout(0.5),
	Dense(512, activation='relu'),
	Dropout(0.5),
	Dense(256, activation='relu'),
	Dropout(0.5),
	Dense(5, activation='softmax')
], output_shape=(5,))

hist = evaluate(model, model_name='mobilenet-binary', num_epochs=10, train_dir='data/proc/binary/train/224/')

pretty_print_history(hist)
