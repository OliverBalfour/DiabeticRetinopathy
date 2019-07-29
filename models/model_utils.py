
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# takes data/vectors/modelname/0-4.npy and creates X and Y vectors
def generate_xy (modelname):
	vdir = 'data/vectors/'+modelname+'/'
	files = os.listdir(vdir)
	if 'X.npy' in files: files.remove('X.npy')
	if 'Y.npy' in files: files.remove('Y.npy')
	ndarrays = [np.load(vdir+fname) for fname in files]
	X = np.concatenate(ndarrays)
	class_ids = [int(fname[0]) for fname in files]
	Y = np.concatenate([np.zeros((len(ndarrays[cid]), 5)) + cid for cid in class_ids])
	for i in range(Y.shape[0]):
		cid = int(Y[i,0])
		Y[i] = np.zeros(5)
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
def process_model (model, modelname, tdir, image_size, preprocess=None, postprocess=None, max_steps=5000):
	for cid in range(5):
		process_class(model, modelname, tdir, 4-cid, image_size, preprocess=preprocess, postprocess=postprocess, max_steps=max_steps)
	generate_xy(modelname)

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
		steps=min(len(generator_flow), max_steps),
		max_queue_size=1
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
