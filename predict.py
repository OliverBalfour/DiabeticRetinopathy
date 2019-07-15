
import pandas as pd
import numpy as np
import os
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

model = keras.models.load_model('models/cnn-final.h5')
model.summary()

print('Testing model now...')

test_dir = 'test_altered/'

# load df and sort alphanumerically
test_df = pd.read_csv('test.csv')
test_df = test_df.sort_values(by=['id_code'])
test_df = test_df.reset_index(drop=True)

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

test_generator = test_datagen.flow_from_directory(
	test_dir,
	target_size=(224, 224),
	batch_size=1,
	class_mode=None,
	shuffle=False # so predictions are in alphanumeric order
)

print('Set up image data generator.')

probabilities = model.predict_generator(test_generator)

print(probabilities)

# can be improved because neighbouring labels are correlated
test_df['diagnosis'] = np.argmax(probabilities, axis=1)
test_df.to_csv('submission.csv', index=False)

# also kaggle version MUST have image transform code embedded; try running preprocessing on train data as part of this file?
# also USE GIT
