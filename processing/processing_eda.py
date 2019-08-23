
import numpy as np
import cv2, os
import matplotlib.pyplot as plt

from image_functions import pre_process, clahe, add_gaussian, post_process

class_names = ['No Retinopathy', 'Mild', 'Moderate', 'Severe', 'Proliferative']
stage_names = ['Raw', 'Resized', 'CLAHE', 'Normalised', 'Clipped']
# class_names = ['Benign', 'Malignant']
classes = np.arange(len(class_names))

def getmat (m, n):
	return np.zeros((m, n), dtype='int').tolist()

# 2d array of 3d tensors (img sizes may differ)
def draw_samples (images, ylabels, xlabels, figsize=(12, 9)):
	fig = plt.figure(figsize=figsize)

	rows = len(images)
	cols = len(images[0])

	for row in range(rows):
		for col in range(cols):
			ax = fig.add_subplot(rows, cols, row * rows + col + 1, xticks=[], yticks=[])
			plt.imshow(images[row][col] / 255)
			if row == 0:
				ax.set_title(xlabels[col])
			if col == 0:
				ax.set_ylabel(ylabels[row])

	plt.subplots_adjust(
		top=0.92,
		bottom=0.05,
		left=0.07,
		right=0.95,
		hspace=0.3,
		wspace=0.2
	)
	plt.show()

proc = [
	lambda read: cv2.cvtColor(cv2.imread(read), cv2.COLOR_BGR2RGB),
	lambda img: cv2.cvtColor(pre_process(img), cv2.COLOR_RGB2BGR),
	lambda img: clahe(img),
	lambda img: add_gaussian(img),
	lambda img: post_process(img, 224)
]


def exemplar_stages ():
	images = getmat(5,5)
	fnames = sorted(os.listdir('images/processing-stages'))
	for row, fname in enumerate(fnames):
		img = 'images/processing-stages/' + fname
		for col, fn in enumerate(proc):
			img = fn(img)
			images[row][col] = img
	draw_samples(images, class_names, stage_names)

# broken atm
def augmentations ():
	images = getmat(2,4)
	fnames = sorted(os.listdir('images/augmentations'))
	for i, fname in enumerate(fnames):
		img = 'images/augmentations/' + fname
		images[i % 2][i // 2] = proc[0](img)
	print(images)
	draw_samples(images, ['']*2, ['']*4, figsize=((12, 4)))

def misclassified ():
	pass

def low_quality_stages ():
	pass


augmentations()
