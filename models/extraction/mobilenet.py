
import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.applications import MobileNet
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input

model = MobileNet(weights='imagenet', include_top=False)

image_size = 224

train_dir = f'data/proc/{str(image_size)}/'
batch_size = 1 # must be 1 for the broken image handling to work nicely

def gen_wrapper (gen, cid):
	while True:
		try:
			yield next(gen)
		except:
			pass

def process_class (cid):
	print('Processing cid ' + str(cid))

	data_generator = ImageDataGenerator(
		rescale=1.0/255,
		preprocessing_function=preprocess_input
	)
	generator_flow = data_generator.flow_from_directory(
		train_dir, target_size=(image_size,image_size), batch_size=batch_size,
		class_mode='categorical', shuffle=False, color_mode='rgb',
		classes=[str(cid)]
	)
	generator = gen_wrapper(generator_flow, cid)
	tensors = model.predict_generator(generator, verbose=1, steps=len(generator_flow), max_queue_size=1)
	print('Generated tensors for cid ' + str(cid))

	# shape is :,7,7,1024 and we want global averages so :,1,1,1024 or equivalently :,1024
	vectors = np.moveaxis(tensors, -1,1).mean(axis=-1).mean(axis=-1)

	np.save('data/vectors/mobilenet'+str(cid)+'.npy', vectors)
	print('Done. Vectors saved')

	del vectors
	del tensors
	del generator
	del generator_flow
	del data_generator

for cid in range(5):
	process_class(4-cid) # backwards
