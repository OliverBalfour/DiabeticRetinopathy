
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2 as MobileNet
from model_utils import SequentialConstructor, evaluate, pretty_print_history

model = SequentialConstructor([
	MobileNet(weights='imagenet', include_top=False),
	GlobalAveragePooling2D(),
	BatchNormalization(),
	Dropout(0.5),
	Dense(512, activation='relu'),
	Dropout(0.5),
	Dense(256, activation='relu'),
	Dropout(0.5)
], output_shape=(2,))

hist = evaluate(model, model_name='mobilenet-binary', train_dir='data/proc/binary/train/224/', num_epochs=10)

pretty_print_history(hist)
