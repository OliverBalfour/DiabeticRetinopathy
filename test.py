
# module to specify base application, number of cut off layers, number of added layers, etc.
# evaluate each application on each image, potentially using data augmentation, and use those as inputs

import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense,GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

# config
batch_size = 12 # don't have the mem for 15
num_epochs = 3
train_dir = 'data/proc/'

#imports the mobilenet model and discards the output layer
base_model = MobileNet(weights='imagenet', include_top=False)

# construct the model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dense(1024, activation='relu')(x)
x = Dense(512, activation='relu')(x)
preds = Dense(5, activation='softmax')(x)

# create the model
model = Model(inputs=base_model.input, outputs=preds)

# specify that we're not continuing to train the preset model
for index, layer in enumerate(model.layers):
	layer.trainable = index > 20

# images must be in class subfolders
# this handles data augmentation
train_datagen = ImageDataGenerator(
	preprocessing_function=preprocess_input,
	rotation_range=20,
	horizontal_flip=True,
	validation_split=0.2 # we need 2 generators
)

# create generators
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

# specify optimiser and loss
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

checkpoint = ModelCheckpoint('models/cnn', save_weights_only=True, verbose=1, period=1)
# model.load_weights('models/cnn')

hist = model.fit_generator(
	generator=train_generator,
	steps_per_epoch=20, #(train_generator.n // batch_size),
	epochs=num_epochs, callbacks=[checkpoint],
	validation_data=valid_generator,
	validation_steps=2 #(valid_generator.n // batch_size)
)

print(hist.history)

model.save('models/cnn-final-2.h5')
# need some kind of save & quit mechanism

#{'loss': [0.8080748800250516, 0.7565884429569193], 'acc': [0.734638, 0.74846864], 'val_loss': [0.8334946387745056, 0.8490050041990489], 'val_acc': [0.73297215, 0.72987616]}
