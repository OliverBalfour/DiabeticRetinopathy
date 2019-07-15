
import numpy as np
import cv2

default_image_size = 224

# scales so that the radius of the image is constant
# approximates the initial radius by calculating the number of non-black pixels along the middle row and halving
def scale_max (img, new_radius):
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

def create_circle_mask (img, rad):
	circle_mask = np.zeros(img.shape)
	# replace pixels in the circle with (1,1,1)
	cv2.circle(circle_mask, (img.shape[1]//2, img.shape[0]//2), rad, (1,1,1), -1)
	return circle_mask

# takes a df row and preprocesses the referenced image
def preprocess (path, id_code, save, rad=default_image_size//2):
	img = cv2.imread(path)
	# scale the image to fit in a 224x224x3 image tensor
	scaled_img = scale_max(img, rad//0.90)
	# recolour the images
	recol_img = cv2.cvtColor(scaled_img, cv2.COLOR_RGB2BGR)
	# gaussian blur adds unwanted border to image, so we need a circular mask
	circle_mask = create_circle_mask(recol_img, rad)
	eye_img = cv2.addWeighted(
		scaled_img, 5, # image with positive weight
		cv2.GaussianBlur(scaled_img, (0,0), (rad//0.9)/50), -5, # blurred image with negative weight
		128 # greyscale scalar
	)
	grey_outline = (1 - circle_mask) * 128
	crop_img = crop_image(eye_img * circle_mask + grey_outline, rad)
	cv2.imwrite(save + id_code + '.png', crop_img)
