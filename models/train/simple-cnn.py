import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import SGD

# config
batch_size = 20
num_epochs = 4
train_dir = 'data/proc/aug/224/'
model_name = 'models/h5/simple-cnn-aug.h5'

# construct the model
# don't put relu on the conv layers!!!!
model = Sequential([
	Input(shape=(224,224,3)),
	Conv2D(16, kernel_size=(7,7)),
	MaxPooling2D(),
	Conv2D(32, kernel_size=(5,5)),
	MaxPooling2D(),
	Flatten(),
	Dense(256, activation='relu'),
	Dense(128, activation='relu'),
	Dense(5, activation='softmax'),
])

train_datagen = ImageDataGenerator(validation_split=0.9)

train_generator = train_datagen.flow_from_directory(
	train_dir, target_size=(224,224), batch_size=batch_size,
	class_mode='categorical', shuffle=True, color_mode='rgb',
	subset='training'
)

valid_generator = train_datagen.flow_from_directory(
	train_dir, target_size=(224,224), batch_size=batch_size,
	class_mode='categorical', shuffle=True, color_mode='rgb',
	subset='validation'
)

def unonehot (labels):
	return np.argmax(labels, axis=1)
def inc (p):
	return np.append(np.zeros(p) + 1, np.zeros(5-p))
def incremental (labels):
	return np.array([inc(np.argmax(labels[y])) for y in range(len(labels))])
def binary (labels):
	return np.array([[0,1,0,0,0] if np.argmax(labels[y]) else [1,0,0,0,0] for y in range(len(labels))])

def wrap (gen):
	while True:
		try:
			data, labels = next(gen)
			yield (data, (labels))
		except GeneratorExit:
			raise GeneratorExit
		except:
			pass

model.compile(
	optimizer='adam',#SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True),
	loss='categorical_crossentropy', metrics=['accuracy']
)

# include wrapping, SGD, dropout, and USE GIT

hist = model.fit_generator(
	generator=(train_generator),
	steps_per_epoch=(train_generator.n // batch_size),
	epochs=num_epochs,
	#validation_data=(valid_generator),
	#validation_steps=(valid_generator.n // batch_size)
)

hist.history.pop('val_loss', None)
hist.history.pop('loss', None)

print(hist.history)

model.save(model_name)
