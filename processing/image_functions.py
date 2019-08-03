
import numpy as np
import cv2


# safe to say we won't want images near 500 pixels large
max_size = 500

# process an image and save it in each specified size
def process_image (read, write, sizes):
	if isinstance(read, str):
		img = cv2.imread(read)
	else:
		img = np.array(read)
	img = process(img)
	for size in sizes:
		cv2.imwrite(
			write.replace('[SIZE]', str(size)),
			post_process(img, size)
		)

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



# DEPRECATED
# takes a bordersize tuple (t,b,l,r) in pixels
# adds grey border
def add_border (img, bordersize):
	return cv2.copyMakeBorder(
		img, borderType=cv2.BORDER_CONSTANT,
		top=bordersize[0], bottom=bordersize[1],
		left=bordersize[2], right=bordersize[3],
		value=(128,128,128)
	)

# DEPRECATED
# crop with bounds checking, if not big enough it adds a grey border
# doesn't like odd sizes
def crop_image_bounds_check (img, size):
	rad = size // 2
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
