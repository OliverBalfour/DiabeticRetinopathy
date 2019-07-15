
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam, SGD, Adagrad, Adadelta, RMSprop
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

import os, cv2
import tensorflow as tf

# load new data
# might need to add: aptos2019-blindness-detection/
comp_dir = '/media/oliver/OliverHDD/Downloads/'
old_dir = '/media/oliver/OliverHDD/Downloads/'
train_df = pd.read_csv(comp_dir + 'train.csv')
train_df['path'] = comp_dir + 'train_images/' + train_df['id_code'] + '.png'
train_df = train_df[[os.path.isfile(i) for i in train_df['path']]]

input_shape = (150, 150, 3)
num_classes = 5

print('loaded df')

images = []
labels = []
def create_training_set(label, path):
	img = cv2.imread(path, cv2.IMREAD_COLOR)
	img = cv2.resize(img, input_shape[:2])
	images.append(np.array(img))
	labels.append(str(label))

for index, sample in train_df.sample(n=1000).iterrows():
	create_training_set(sample['diagnosis'], sample['path'])
print('Finished loading train set')

Y = to_categorical(labels)
X = np.array(images)
X = X / 255

X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=0.2, random_state=22)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

feat_extraction = ImageDataGenerator(
	featurewise_center=True,
	featurewise_std_normalization=True,
	rotation_range=20,
	width_shift_range=0.2,
	height_shift_range=0.2,
	horizontal_flip=True
)

feat_extraction.fit(X_train)

model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.summary()


model.compile(optimizer= Adam(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
batch_size = 15
epochs = 1

from tensorflow.keras.callbacks import ModelCheckpoint

check = ModelCheckpoint(filepath='cnn.h5', save_best_only=True, verbose=1)

model.fit_generator(
	feat_extraction.flow(X_train, Y_train, batch_size=batch_size),
	epochs=epochs, validation_data=(X_valid, Y_valid), callbacks=[check],
	steps_per_epoch=(X_train.shape[0] // batch_size), verbose=1
)

# TODO: ensure the directories are correct
test_df = pd.read_csv(comp_dir + 'test.csv')
test_df['path'] = comp_dir + 'test_images/' + test_df['id_code'] + '.png'
test_df = test_df[[os.path.isfile(i) for i in test_df['path']]]

images = []
def create_test_set(path):
	img = cv2.imread(path, cv2.IMREAD_COLOR)
	img = cv2.resize(img, input_shape)
	images.append(np.array(img))

for index, sample in test_df.iterrows():
	create_test_set(sample['path'])
print('Finished loading test set')

X_test = np.array(images)
X_test = X_test / 255
Y_test = model.predict(X_test)

test_df['diagnosis'] = [str(x) for x in np.argmax(Y_test, axis=0)]
test_df.to_csv('submission.csv')
