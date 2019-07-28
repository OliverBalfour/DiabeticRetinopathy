
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

batch_size = 10
num_epochs = 5
train_dir = 'data/proc/aug/224/'
model_name = 'models/h5/mobilenet-headless.h5'

image_size = 224
image_shape = (image_size,image_size,3)

model = Sequential([
	MobileNetV2(weights='imagenet', include_top=False, input_shape=image_shape),
	GlobalAveragePooling2D(),
	BatchNormalization(),
	Dropout(0.5),
	Dense(512, activation='relu'),
	Dropout(0.5),
	Dense(256, activation='relu'),
	Dropout(0.5),
	Dense(5, activation='softmax')
])

def getsubset (gen, subset):
	return train_datagen.flow_from_directory(
		train_dir, target_size=(image_size,image_size), batch_size=batch_size,
		class_mode='categorical', shuffle=True, color_mode='rgb',
		subset=subset
	)

train_datagen = ImageDataGenerator(validation_split=0.1, rescale=1.0/255)
train_generator = getsubset(train_datagen, 'training')
valid_generator = getsubset(train_datagen, 'validation')

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

hist = model.fit_generator(
	generator=train_generator, steps_per_epoch=(train_generator.n // batch_size),
	validation_data=valid_generator, validation_steps=(valid_generator.n // batch_size),
	epochs=num_epochs
)

print(hist.history)
print('Training Accuracy:   ' + str(round(hist.history['acc'][-1]*100,2))+'%')
print('Validation Accuracy: ' + str(round(hist.history['val_acc'][-1]*100,2))+'%')

model.save(model_name)
