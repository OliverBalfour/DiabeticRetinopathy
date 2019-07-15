
import numpy as np
import pandas as pd
import cv2, os

df_train_1 = pd.read_csv('train.csv')
df_train_1['path'] = 'train_images/' + df_train_1.id_code + '.png'

df_train_2 = pd.read_csv('trainLabels_cropped.csv')
df_train_2['id_code'] = df_train_2.image
df_train_2['diagnosis'] = df_train_2.level
df_train_2['path'] = 'resized_train_cropped/resized_train_cropped/' + df_train_2.id_code + '.jpeg'
df_train_2.drop(['Unnamed: 0', 'Unnamed: 0.1', 'image', 'level'],inplace=True,axis=1)
df_train_2 = df_train_2[[os.path.isfile(fname) for fname in df_train_2.path]]

df_train = pd.concat([df_train_2, df_train_1])

df_test = pd.read_csv('test.csv')
df_test['path'] = 'test_images/' + df_test.id_code + '.png'

image_size = 224

# scales so that the radius of the image is constant
# approximates the initial radius by calculating the number of non-black pixels along the middle row and halving
def scale_max (img, new_radius):
	middle_row = img[len(img) // 2].sum(0) # sum colour channels across middle row
	threshold = middle_row.mean() / 8 # color threshold for sum of colour channels
	radius = np.sum(middle_row > threshold) / 2
	return cv2.resize(img, (0,0), fx=(new_radius / radius), fy=(new_radius / radius))

# takes a bordersize tuple (t,b,l,r) in pixels
# adds grey border
def add_border (img, bordersize):
	return cv2.copyMakeBorder(
		img, borderType=cv2.BORDER_CONSTANT,
		top=bordersize[0], bottom=bordersize[1],
		left=bordersize[2], right=bordersize[3],
		value=(128,128,128)
	)

# crop with bounds checking, if not big enough it adds grey border
def crop_image (img, rad):
	mx, my = (img.shape[1]//2, img.shape[0]//2)
	# bounds checking
	if mx - rad < 0:
		new_img = add_border(img, (0,0,rad-mx,rad-mx))
		return crop_image(new_img, rad)
	if my - rad < 0:
		new_img = add_border(img, (rad-my,rad-my,0,0))
		return crop_image(new_img, rad)
	# crop and return
	return img[my-rad:my+rad, mx-rad:mx+rad]

# takes a df row and preprocesses the referenced image
def preprocess (row, save, rad):
	a = cv2.imread(row.path)
	a = scale_max(a, rad//0.9)
	mx, my = (a.shape[1]//2, a.shape[0]//2)
	b = np.zeros(a.shape)
	cv2.circle(b, (mx, my), rad, (1,1,1), -1, 8, 0)
	aa = cv2.addWeighted(a, 4, cv2.GaussianBlur(a, (0,0), (rad//0.9)/30), -4, 128) * b + 128 * (1 - b)
	crop_img = crop_image(aa, rad)
	cv2.imwrite(save + row.id_code + '.png', crop_img)

for row in df_train.itertuples():
	cls = str(row.diagnosis)
	try:
		preprocess(row, f'train_altered/{cls}/', image_size//2)
	except:
		print(row.path)

print('Preprocessed training images.')

for row in df_test.itertuples():
	preprocess(row, 'test_altered/data/', image_size//2) # needs a subdir for class_mode=None to work

print('Preprocessed testing images.')
