
import numpy as np
import os
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.densenet import preprocess_input # imagenet processing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense


### MODEL RELATED

image_shape = (224,224,3)

# takes compiled model, trains it, and saves it and returns history
# if a small number of epochs is used models may be loaded and re-evaluated as needed
def evaluate (
		model,
		model_name='model-'+str(np.random.randint(1e8,1e9)),
		train_dir=None,
		num_epochs=15, batch_size=10,
		image_shape=image_shape,
		preprocessing_function=keras.applications.densenet.preprocess_input
	):

	train_datagen = ImageDataGenerator(validation_split=0.1, rescale=1.0/255, preprocessing_function=preprocessing_function)
	train_generator = getsubset(train_datagen, 'training', train_dir, image_shape[0], batch_size)
	valid_generator = getsubset(train_datagen, 'validation', train_dir, image_shape[0], batch_size)

	history = model.fit_generator(
		generator=train_generator, steps_per_epoch=(train_generator.n // batch_size),
		validation_data=valid_generator, validation_steps=(valid_generator.n // batch_size),
		epochs=num_epochs
	)

	model.save('models/h5/'+model_name+'.h5')

	return history.history

# wraps a sequential model with the correct input and output shapes and compiles
def SequentialConstructor (
		arr, input_shape=image_shape, output_shape=(5,),
		optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']
	):
	model = Sequential(
		[Input(shape=input_shape)] + flatten(arr) + [Dense(output_shape[0], activation='softmax')]
	)
	model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

	return model

def flatten (arr):
	newarr = []
	for el in arr:
		if type(el) is list or type(el) is tuple:
			for t in flatten(el):
				newarr.append(t)
		else:
			newarr.append(el)
	return newarr

def pretty_print_history (hist):
	print(hist)
	print('Training Accuracy:   ' + str(round(hist['acc'][-1]*100,2))+'%')
	print('Validation Accuracy: ' + str(round(hist['val_acc'][-1]*100,2))+'%')
	with open('tmp', 'w+') as f:
		f.write(str(hist))

# get generator for training/validation subset of ImageDataGenerator
def getsubset (gen, subset, directory, image_size, batch_size):
	return gen.flow_from_directory(
		directory, target_size=(image_size,image_size), batch_size=batch_size,
		class_mode='categorical', shuffle=True, color_mode='rgb',
		subset=subset
	)






### VECTOR RELATED


# takes data/vectors/modelname/0-4.npy and creates X and Y vectors
def generate_xy (modelname, binary=False):
	vdir = 'data/vectors/'+modelname+'/'
	files = os.listdir(vdir)
	if 'X.npy' in files: files.remove('X.npy')
	if 'Y.npy' in files: files.remove('Y.npy')
	ndarrays = [np.load(vdir+fname) for fname in files]
	X = np.concatenate(ndarrays)
	class_ids = [int(fname[0]) for fname in files]
	Y = np.concatenate([np.zeros((len(ndarrays[cid]), 5 if not binary else 2)) + cid for cid in class_ids])
	for i in range(Y.shape[0]):
		cid = int(Y[i,0])
		Y[i] = np.zeros(5 if not binary else 2)
		Y[i,cid] = 1
	np.save(vdir+'X.npy', X)
	np.save(vdir+'Y.npy', Y)

def load_xy (modelname):
	X = np.load('data/vectors/'+modelname+'/X.npy')
	Y = np.load('data/vectors/'+modelname+'/Y.npy')
	return (X, Y)

# wraps a generator and ignores errors
def gen_wrapper (gen):
	while True:
		try:
			yield next(gen)
		except GeneratorExit:
			raise GeneratorExit
		except:
			pass

# process all classes and generate X, Y
def process_model (model, modelname, tdir, image_size, preprocess=None, postprocess=None, max_steps=5000, binary=False):
	for cid in range(5 if not binary else 2):
		process_class(model, modelname, tdir, cid, image_size, preprocess=preprocess, postprocess=postprocess, max_steps=max_steps)
	generate_xy(modelname, binary=binary)

# processes a class on a model and saves to data/vectors/model/0-4.npy
# data processing must be EXACTLY THE SAME as in training: abstract away image loading and processing perhaps?
def process_class (model, modelname, tdir, cid, image_size, preprocess=None, postprocess=None, max_steps=None):
	if preprocess is None:
		preprocess = lambda x: x

	print('Processing cid ' + str(cid))

	data_generator = ImageDataGenerator(
		rescale=1.0/255,
		preprocessing_function=preprocess
	)
	generator_flow = data_generator.flow_from_directory(
		tdir, target_size=(image_size,image_size), batch_size=1, # must be 1 for the broken image handling to work nicely
		class_mode='categorical', shuffle=True, color_mode='rgb',
		classes=[str(cid)]
	)
	generator = gen_wrapper(generator_flow)
	tensors = model.predict_generator(
		generator, verbose=1,
		steps=max_steps,
		#max_queue_size=1
	)

	if postprocess is not None:
		tensors = postprocess(tensors)
	np.save('data/vectors/'+modelname+'/'+str(cid)+'.npy', tensors)
	del tensors

# alphanumerically sorts all files in data/proc/224 and returns the diagnoses for each
def get_sorted_classes ():
	train_dir = f'data/proc/224/'
	files = []
	for cid in range(5):
		for x in os.listdir(train_dir + str(cid)):
			files.append(x+str(cid))

	files.sort()
	classes = [int(x[-1]) for x in files]
	return classes

# alphanumerically sorts all files in data/proc/224 and returns their their dataset
# True is new, old is False
def get_sorted_datasources ():
	train_dir = f'data/proc/224/'
	files = []
	for cid in range(5):
		for x in os.listdir(train_dir + str(cid)):
			files.append(x+str(cid))

	files.sort()
	datasources = [1 - (x.endswith('left.png') or x.endswith('right.png')) for x in files]
	return datasources


def process_model_without_saving (model, modelname, tdir, image_size, preprocess=None, postprocess=None, max_steps=5000, binary=False):
	data = []
	for cid in range(2):
		print('Processing cid ' + str(cid))
		data_generator = ImageDataGenerator(
			rescale=1.0/255,
			preprocessing_function=preprocess
		)
		generator_flow = data_generator.flow_from_directory(
			tdir, target_size=(image_size,image_size), batch_size=1,
			class_mode='categorical', shuffle=True, color_mode='rgb',
			classes=[str(cid)]
		)
		generator = gen_wrapper(generator_flow)
		tensors = model.predict_generator(
			generator, verbose=1,
			steps=max_steps,
		)
		if postprocess is not None:
			tensors = postprocess(tensors)
		data.append(tensors)
	X = np.concatenate(data[0], data[1])
	cids = np.concatenate(np.zeros(len(data[0])), np.zeros(len(data[1])) + 1)
	Y = np.identity(2)[cids]
	return X, Y
