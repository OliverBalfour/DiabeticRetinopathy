
# module to specify base application, number of cut off layers, number of added layers, etc.
# evaluate each application on each image, potentially using data augmentation, and use those as inputs
# change methodology to support arbitrary class groupings and different last layer encodings (not one-hot)
# optimised quad weighted kappa and show different, more useful metrics (sensitivity/specificity and conf mats)

import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# config
batch_size = 12 # don't have the mem for 15
num_epochs = 3
train_dir = 'data/proc/'
model_name = 'models/h5/mobilenet-transfer.h5'

#imports the mobilenet model and discards the output layer
base_model = MobileNet(weights='imagenet', include_top=False)

# construct the model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dense(256, activation='relu')(x)
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

hist = model.fit_generator(
	generator=train_generator,
	steps_per_epoch=20, #(train_generator.n // batch_size),
	epochs=num_epochs,
	validation_data=valid_generator,
	validation_steps=2 #(valid_generator.n // batch_size)
)

print(hist.history)

model.save(model_name)
