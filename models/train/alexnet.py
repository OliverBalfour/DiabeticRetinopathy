
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization, Conv2D, MaxPooling2D, Flatten
from model_utils import SequentialConstructor, evaluate, pretty_print_history

# config
batch_size = 128
num_epochs = 5 # 10mins each
train_dir = 'data/proc/aug/224/'
model_name = 'models/h5/alexnet-aug-1.h5'

# transform x or (x,y) to (x,x) or (x,y) cause I'm lazy
def shapify (data):
	if type(data) is list or type(data) is tuple:
		return data
	else:
		return (data, data)

def Conv2DLayer (filters=None, input_shape=None, kernel_size=None, strides=1, pool_size=2, pool_strides=2, activation=None, batchnorm=False, pooling=True):
	layers = [Conv2D(filters=filters, input_shape=input_shape, kernel_size=shapify(kernel_size), strides=shapify(strides), activation=activation)]
	if pooling:
		layers.append(MaxPooling2D(pool_size=shapify(pool_size), strides=shapify(pool_strides)))
	if batchnorm:
		layers.append(BatchNormalization())
	return layers

def DenseLayer (units, input_shape=None, activation='relu', dropout=0, batchnorm=False):
	layers = [Dense(units, input_shape=input_shape, activation=activation)]
	if dropout:
		layers.append(Dropout(dropout))
	if batchnorm:
		layers.append(BatchNormalization())
	return layers

model = SequentialConstructor([
	*Conv2DLayer(filters=96, kernel_size=11, strides=4, batchnorm=True),
	*Conv2DLayer(filters=256, kernel_size=11,strides=1, batchnorm=True),
	*Conv2DLayer(filters=384, kernel_size=3, strides=1, pooling=False, batchnorm=True),
	*Conv2DLayer(filters=384, kernel_size=3, strides=1, pooling=False, batchnorm=True),
	*Conv2DLayer(filters=384, kernel_size=3, strides=1, pooling=False, batchnorm=True),
	*Conv2DLayer(filters=256, kernel_size=3, strides=1, batchnorm=True),
	Flatten(),
	*DenseLayer(4096, activation='relu', dropout=0.4, batchnorm=True, input_shape=(224*224*3,)),
	*DenseLayer(2048, activation='relu', dropout=0.4, batchnorm=True),
	*DenseLayer(1024, activation='relu', dropout=0.4),
	*DenseLayer(512, activation='relu')
], output_shape=(2,))

hists = []
iteration = 1
num_epochs = 5
while input(f'Train for {str(num_epochs)} more epochs? y/n: ') == 'y':
	hist = evaluate(model, model_name='olivernet-binary-'+str(iteration*num_epochs), train_dir='data/proc/binary/train/224/', num_epochs=num_epochs)
	hists.append(hist)
	iteration += 1

for hist in hists:
	pretty_print_history(hist)

#if input('plot? y/n: ') == 'y':
#	import matplotlib.pyplot as plt
#	plt.plot(np.arange(num_epochs), hist['acc'], 'r-')
#	plt.plot(np.arange(num_epochs), hist['val_acc'], 'b-')
#	plt.show()
