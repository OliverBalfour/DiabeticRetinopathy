
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

# resize to have a certain radius
def scale_radius (img, scale):
	x = img[img.shape[0]//2,:,:].sum(1)
	r = (x > x.mean() / 10).sum() / 2
	s = scale / r
	return cv2.resize(img, (0,0), fx=s, fy=s)

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

# takes a df row and preprocesses the referenced image
def preprocess (row, save):
	# print(row.path)
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
	fname = save + row.id_code + '.png'
	cv2.imwrite(fname, crop_img)

for row in df_train.itertuples():
	cls = str(row.diagnosis)
	try:
		preprocess(row, f'train_altered/{cls}/')
	except:
		print(row.path)

print('Preprocessed training images.')

for row in df_test.itertuples():
	preprocess(row, 'test_altered/data/') # needs a subdir for class_mode=None to work

print('Preprocessed testing images.')

