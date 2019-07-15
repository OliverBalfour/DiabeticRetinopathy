
import numpy as np
import pandas as pd
import cv2, os, shutil
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

input_dir = '../input/aptos2019-blindness-detection/'
test_dir = 'test_images/'
os.mkdir(test_dir)
os.mkdir(test_dir + 'data/')

### DATA PROCESSING

df_test = pd.read_csv(input_dir + 'sample_submission.csv')
df_test['path'] = input_dir + 'test_images/' + df_test.id_code + '.png'

# resize to have a certain radius
def scale_radius (img, new_radius):
	middle_row = img[len(img) // 2].sum(1) # sum colour channels across middle row
	threshold = middle_row.mean() / 8 # color threshold for sum of colour channels
	radius = np.sum(middle_row > threshold) / 2
	# exceptionally dark images may evaluate as having tiny radii
	if radius < len(img) // 3:
		radius = min(img.shape[:2]) // 3
	return cv2.resize(img, (0,0), fx=(new_radius / radius), fy=(new_radius / radius))

# takes a bordersize tuple (t,b,l,r) in pixels
# adds grey border
def add_border (img, bordersize):
	return cv2.copyMakeBorder(
		img, borderType=cv2.BORDER_CONSTANT,
		top=bordersize[0], bottom=bordersize[1],
		left=bordersize[2], right=bordersize[3],
		value=(127,127,127)
	)

# crop with bounds checking, if not big enough it adds grey border
def crop_image (img, midpoint, rad):
	mx, my = midpoint
	# bounds checking
	if mx - rad < 0:
		width = rad - mx
		new_img = add_border(img, (0,0,width,width))
		return crop_image(new_img, (new_img.shape[1]//2, new_img.shape[0]//2), rad)
	if my - rad < 0:
		height = rad - my
		new_img = add_border(img, (height,height,0,0))
		return crop_image(new_img, (new_img.shape[1]//2, new_img.shape[0]//2), rad)
	# crop and return, indexed as y then x
	return img[my-rad:my+rad, mx-rad:mx+rad]

for row in df_test.itertuples():
	a = cv2.imread(row.path)
	# scale to radius
	scale = 120
	a = scale_radius(a, scale)
	# add circle mask
	mx, my = (a.shape[1]//2, a.shape[0]//2)
	b = np.zeros(a.shape)
	cv2.circle(b, (mx, my), int(scale*0.9), (1,1,1), -1, 8, 0)
	# cool transform
	aa = cv2.addWeighted(a, 4, cv2.GaussianBlur(a, (0,0), scale/30), -4, 128) * b + 128 * (1 - b)
	half = 112 # 224 // 2
	# crop
	crop_img = crop_image(aa, (mx, my), half)
	# save
	fname = test_dir + 'data/' + row.id_code + '.png' # needs a subdir for class_mode=None to work
	cv2.imwrite(fname, crop_img)
	del a
	del aa
	del crop_img

print('Preprocessed testing images. Loading model...')


### PREDICTIONS

model = keras.models.load_model('../input/v1model/cnn-final.h5')
model.summary()

print('Testing model now...')

# load df and sort alphanumerically
test_df = pd.read_csv(input_dir + 'test.csv')
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

# can be improved because neighbouring labels are correlated
test_df['diagnosis'] = np.argmax(probabilities, axis=1)
test_df.to_csv('submission.csv', index=False)

print('Deleting unneeded files...')

shutil.rmtree(test_dir)
print('Done! Saved to submission.csv')
