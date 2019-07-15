
# module to specify base application, number of cut off layers, number of added layers, etc.
# evaluate each application on each image, potentially using data augmentation, and use those as inputs
# change methodology to support arbitrary class groupings and different last layer encodings (not one-hot)
# optimised quad weighted kappa and show different, more useful metrics (sensitivity/specificity and conf mats)

import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# config
batch_size = 10 # don't have the mem for 15
num_epochs = 3
train_dir = 'data/proc/'
model_name = 'models/h5/simple-cnn.h5'

# construct the model
model_in = Input(shape=(224,224,3))
x = Conv2D(32, kernel_size=(5,5), activation='relu')(model_in)
x = MaxPooling2D(pool_size=(2,2), strides=(2,2))(x)
x = Conv2D(64, kernel_size=(5,5), activation='relu')(x)
x = MaxPooling2D(pool_size=(2,2), strides=(2,2))(x)
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = Dense(256, activation='relu')(x)
model_out = Dense(5, activation='softmax')(x)

# create the model
model = Model(inputs=model_in, outputs=model_out)

train_datagen = ImageDataGenerator(validation_split=0.2)

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
