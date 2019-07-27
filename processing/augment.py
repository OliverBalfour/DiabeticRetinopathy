
import numpy as np
import pandas as pd
import cv2, os, random
import imgaug as ia
import imgaug.augmenters as iaa
from image_functions import process_image
from dataframes import get_train_df_old, get_train_df_new

train_df_new = get_train_df_new()
train_df_old = get_train_df_old()
sizes = (224,) #299)
class_size = 1000 # 5K images overall


def map_df (df, dirname):
	mappings = [[] for cid in range(5)] # class indexed list of (source, dest)
	for row in df.itertuples():
		mappings[row.diagnosis].append((row.path, f'{dirname}[SIZE]/{str(row.diagnosis)}/{row.id_code}.png'))
	return mappings

# augmenting includes rotating, flipping, changing colors, and blurring a little

seq = iaa.Sequential([
	iaa.Fliplr(0.5),
	iaa.Flipud(0.5),
	iaa.Affine(rotate=(-45, 45)),
	iaa.Sometimes(0.25, iaa.GaussianBlur(sigma=(0, 0.25))),
	iaa.Sometimes(0.25, iaa.ContrastNormalization((0.8, 1.2))),
	iaa.Multiply((0.5, 1.4))
], random_order=True)

# todo: optimise so that it uses batches (perhaps do all epochs at once, and process augment the same image multiple times?)
def augment (source):
	img = np.array(cv2.cvtColor(cv2.imread(source), cv2.COLOR_BGR2RGB), dtype=np.uint8)
	return seq.augment_image(img)

# augment and process each sample once
def epoch (samples, stop=None):
	stop = len(samples) if stop is None else stop
	for i, sample in enumerate(samples):
		if i >= stop:
			break
		# process_image takes source file name or np.array
		fname = sample[1][:-4] + '_aug_' + str(random.randint(1e6,1e7)) + '.png'
		process_image(augment(sample[0]), fname, sizes)
		print('.', end='', flush=True)

# samples should be shuffled
def process_class_augmentations (samples, class_size):
	# copy over max(class_size, len(samples)) images
	for sample in samples[:class_size]:
		if not os.path.isfile(sample[0]):
			process_image(sample[0], sample[1], sizes)
			print('.', end='', flush=True)

	# augment images if needed
	if len(samples) < class_size:
		needed = class_size - len(samples)
		# loop for each time that all image augs are needed, then randomly select the remainder
		for _ in range(needed // len(samples)):
			epoch(samples)
		epoch(samples, stop=(needed % len(samples)))

def generate_augmentations (mapping, class_size=class_size):
	for cid, samples in enumerate(mapping):
		samples = samples.copy()
		random.shuffle(samples)
		process_class_augmentations(samples, class_size)

generate_augmentations(map_df(train_df_new, 'data/proc/aug/'))
# generate_augmentations(map_df(train_df_old, 'data/aug_old/'))

print('\nAugmented training images.')
