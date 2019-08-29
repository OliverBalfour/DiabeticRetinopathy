
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
def draw_samples (images, ylabels, xlabels, figsize=(12, 9), **kwargs):
	fig = plt.figure(figsize=figsize)

	rows = len(images)
	cols = len(images[0])

	for row in range(rows):
		for col in range(cols):
			# print(row * rows + col + 1)
			ax = fig.add_subplot(rows, cols, row * cols + col + 1, xticks=[], yticks=[])
			plt.imshow(images[row][col] / 255)
			if row == 0:
				ax.set_title(xlabels[col])
			if col == 0:
				ax.set_ylabel(ylabels[row])
	plt.subplots_adjust(**kwargs)
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
	draw_samples(
		images, class_names, stage_names,
		top=0.92,
		bottom=0.05,
		left=0.07,
		right=0.95,
		hspace=0.3,
		wspace=0.2
	)

def augmentations ():
	images = getmat(1,4)
	fnames = sorted(os.listdir('images/augmentations'))
	for x, fname in enumerate(fnames):
		images[0][x] = proc[0](f'images/augmentations/{fname}')
	draw_samples(
		images, [''], ['Original', 'Augmentation 1', 'Augmentation 2', 'Augmentation 3'], figsize=((16, 4.5)),
		top=0.93,
		bottom=0.05,
		left=0.03,
		right=0.97,
		hspace=0.3,
		wspace=0.1
	)

def misclassified ():
	fnames = sorted(os.listdir('images/bad-labels'))
	images = getmat(2, len(fnames)//2)
	fnames = [[fnames[i], fnames[i+1]] for i in range(0, len(fnames), 2)]
	for x, col in enumerate(fnames):
		for y, fname in enumerate(col):
			images[y][x] = proc[0]('images/bad-labels/' + fname)
	draw_samples(
		images, ['Original', 'Processed'], ['No Retinopathy', 'No Retinopathy', 'Proliferative'], figsize=((12, 6)),
		top=0.92,
		bottom=0.05,
		left=0.07,
		right=0.95,
		hspace=0.1,
		wspace=0.2
	)

def low_quality_stages ():
	fnames = sorted(os.listdir('images/overunder'))
	cols = len(fnames)//2
	images = getmat(2, cols)
	fnames = [[fnames[i], fnames[i+1]] for i in range(0, len(fnames), 2)]
	for x, col in enumerate(fnames):
		for y, fname in enumerate(col):
			images[y][x] = proc[0]('images/overunder/' + fname)
	draw_samples(
		images, ['Original', 'Processed'], ['']*cols, figsize=((16, 4)),
		top=0.95,
		bottom=0.05,
		left=0.03,
		right=0.98,
		hspace=0.05,
		wspace=0.2
	)


exemplar_stages()
augmentations()
misclassified()
low_quality_stages()
