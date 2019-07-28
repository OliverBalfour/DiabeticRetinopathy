import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import SGD
import os

batch_size = 20
num_epochs = 5
train_dir = 'data/proc/aug/224/'
model_name = 'models/h5/simple-cnn-aug-2.h5'

model = Sequential([
	Input(shape=(224,224,3)),
	Conv2D(96, kernel_size=(7,7)),
	MaxPooling2D(),
	Conv2D(64, kernel_size=(5,5)),
	MaxPooling2D(),
	Flatten(),
	Dense(512, activation='relu'),
	Dropout(0.4),
	Dense(256, activation='relu'),
	Dropout(0.4),
	Dense(5, activation='relu')
])

train_datagen = ImageDataGenerator(validation_split=0.8)

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
			yield (data, incremental(labels))
		except GeneratorExit:
			raise GeneratorExit
		except:
			pass

model.compile(
	optimizer='adam',#SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True),
	loss='categorical_crossentropy', metrics=['accuracy']
)

hist = model.fit_generator(
	generator=wrap(train_generator),
	steps_per_epoch=(train_generator.n // batch_size),
	epochs=num_epochs,
#	validation_data=wrap(valid_generator),
#	validation_steps=(valid_generator.n // batch_size)
)

hist.history.pop('val_loss', None)
hist.history.pop('loss', None)

print(hist.history)

print('Train Acc:  ' + str(hist.history['acc'][-1]))
if hasattr(hist.history, 'val_acc'): print('Valid Acc:  ' + str(hist.history['val_acc'][-1]))
num = [len(os.listdir(train_dir + str(cid))) for cid in range(5)]
print('Random Acc: ' + str(max(num)/np.sum(num)))
