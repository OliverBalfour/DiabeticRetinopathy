
import numpy as np
import cv2

from image_functions import crop_image, circle_clip, clahe, resize_min, scale_eye_diameter, add_gaussian

# safe to say we won't want images near 500 pixels large
max_size = 500

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
	img = resize_min(img, int(size*1.1))
	img = crop_image(img, size)
	img = circle_clip(img)
	return img
