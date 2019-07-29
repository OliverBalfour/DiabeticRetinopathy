
import numpy as np
import pandas as pd
import cv2, os, shutil
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

input_dir = '../input/aptos2019-blindness-detection/'
test_dir = 'test_images/'
os.mkdir(test_dir)
os.mkdir(test_dir + 'data/')

### DATA PROCESSING

# load df and sort alphanumerically
test_df = pd.read_csv(input_dir + 'test.csv')
test_df = test_df.sort_values(by=['id_code'])
test_df = test_df.reset_index(drop=True)
test_df['path'] = input_dir + 'test_images/' + test_df.id_code + '.png'


### non kaggle specific

# safe to say we won't want images anywhere near 500 pixels large
# (resizes to 500px before processing to speed up)
max_size = 500
sizes = (224,)

# make the image much smaller to speed up processing, and recolor it
def pre_process (img):
	return cv2.cvtColor(
		resize_min(img, max_size),
		cv2.COLOR_BGR2RGB
	)

# preprocess, apply CLAHE then Gaussian normalisation
# but don't postprocess (multiple postprocesses of the same image may be required)
def process (img):
	return add_gaussian(clahe(pre_process(img)))

# clip, resize and crop the image to the specified size (square)
def post_process (img, size):
	img = scale_eye_diameter(img, max_size // 2)
	img = resize_min(img, int(size*1.2))
	img = crop_image(img, size)
	img = circle_clip(img)
	return img

def crop_image (img, size):
	mx, my = (img.shape[1]//2, img.shape[0]//2)
	rad = size // 2
	if size % 2 == 0:
		return img[my-rad:my+rad, mx-rad:mx+rad]
	else:
		return img[my-rad:my+rad+1, mx-rad:mx+rad+1]

# clips image to a circle (not necessarily a square), replacing outside with grey
def circle_clip (img):
	circle_mask = np.zeros(img.shape)
	# replace pixels in the circle with (1,1,1)
	cv2.circle(circle_mask, (img.shape[1]//2, img.shape[0]//2), min(img.shape[:2])//2, (1,1,1), -1)
	# create outline
	grey_outline = (1 - circle_mask) * 128
	# clip to circle
	return img * circle_mask + grey_outline

# apply Contrast Limited Adaptive Histogram Equalisation (CLAHE)
def clahe (img):
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	for i in range(3):
		img[:,:,i] = clahe.apply(img[:,:,i])
	return img

def add_gaussian (img):
	return cv2.addWeighted(
		img, 5, # image with positive weight
		cv2.GaussianBlur(img, (0,0), 3), -5, # blurred image with negative weight
		128 # greyscale scalar
	)

# resize an image to the smallest image possible such that min(img.shape[:2]) == min
def resize_min (img, min):
	h, w = img.shape[:2]
	if h == w:
		return cv2.resize(img, (min, min))
	elif h < w:
		return cv2.resize(img, (int(w * (min / h)), min))
		return cv2.resize(img, (min, min))
	elif w < h:
		return cv2.resize(img, (min, int(h * (min / w))))

# scales to a new diameter (approximates the initial radius using primitive edge detection along the middle row)
def scale_eye_diameter (img, diameter):
	middle_row = img[len(img) // 2].sum(1) # sum colour channels across middle row
	diff = np.abs(np.diff(middle_row))
	edges = [i for i,x in enumerate(diff > diff.mean()) if x]
	if len(edges) > 2:
		radius = (edges[-1] - edges[0]) / 2
	else:
		radius = np.sum(img.shape[:2]) // 4
	# exceptionally dark images may evaluate as having tiny radii
	if radius < min(img.shape[:2]) / 2:
		radius = min(img.shape[:2]) // 2
	if radius > (max(img.shape[:2]) + min(img.shape[:2])) / 4:
		radius = (max(img.shape[:2]) + min(img.shape[:2])) // 4
	return cv2.resize(img, (0,0), fx=((diameter / 2) / radius), fy=((diameter / 2) / radius))

### end non kaggle specific

print('Preprocessing testing images')

# process an image and save it in each specified size
def process_image (read, write, sizes):
	if isinstance(read, str):
		img = cv2.imread(read)
	else:
		img = np.array(read)
	img = process(img)
	for size in sizes:
		cv2.imwrite(write, post_process(img, size))

def iterate_df (df, dirname):
	for row in df.itertuples():
		try:
			process_image(row.path, f'{dirname}/{row.id_code}.png', sizes)
		except Exception as err:
			print(row.path)
			print(err)

iterate_df(test_df, test_dir+'data/')

### PREDICTIONS

print('Loading model from H5 and creating ImageDataGenerator')

model = keras.models.load_model('../input/v2model/densenet121-categorical.h5')
model.summary()


test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

test_generator = test_datagen.flow_from_directory(
	test_dir,
	target_size=(224, 224),
	batch_size=1,
	class_mode=None,
	shuffle=False # so predictions are in alphanumeric order
)

print('Computing predictions')

# could corruption cause mismatches?
probabilities = model.predict_generator(test_generator)

print('Saving predictions and cleaning filesystem')

# can be improved because neighbouring labels are correlated
test_df['diagnosis'] = np.argmax(probabilities, axis=1)
test_df.to_csv('submission.csv', index=False)
shutil.rmtree(test_dir)

print('Done!')
