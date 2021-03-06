
import numpy as np
import pandas as pd
import cv2, os, random
import imgaug as ia
import imgaug.augmenters as iaa
from image_functions import process_image

def map_df (df, dirname, binary=False):
	mappings = [[] for cid in range(5)] # class indexed list of (source, dest)
	for row in df.itertuples():
		if binary:
			folder = '0' if row.diagnosis == 0 else '1'
		else:
			folder = str(row.diagnosis)
		mappings[int(folder)].append((row.path, f'{dirname}[SIZE]/{folder}/{row.id_code}.png'))
	return mappings

# augmenting includes rotating, flipping, changing colors, and blurring a little
# rotations exclude 20 degrees either side of the image
def get_augment_seq ():
	return iaa.Sequential([
		iaa.Fliplr(0.5),
		iaa.Flipud(0.5),
		iaa.Affine(rotate=(20, -20), scale=(1.0, 1.2)),
		iaa.Sometimes(0.25, iaa.GaussianBlur(sigma=(0, 0.5))),
		iaa.ContrastNormalization((0.9, 1.1)),
		iaa.Multiply((0.5, 1.4))
	], random_order=True)

# todo: optimise so that it uses batches (perhaps do all epochs at once, and process augment the same image multiple times?)
def augment (source_file, seq):
	img = np.array(cv2.cvtColor(cv2.imread(source_file), cv2.COLOR_BGR2RGB), dtype=np.uint8)
	return seq.augment_image(img)

# augment and process each sample once
def epoch (samples, seq, sizes, stop=None):
	stop = len(samples) if stop is None else stop
	for i, sample in enumerate(samples):
		if i >= stop:
			break
		# process_image takes source file name or np.array
		fname = sample[1][:-4] + '_aug_' + str(random.randint(1e6,1e7)) + '.png'
		process_image(augment(sample[0], seq), fname, sizes)
		# print('.', end='', flush=True)

# samples should be shuffled
def process_class_augmentations (samples, class_size, seq, sizes):
	if len(samples) == 0: return
	# copy over max(class_size, len(samples)) images
	for sample in samples[:class_size]:
		if not os.path.isfile(sample[1]):
			process_image(sample[0], sample[1], sizes)
			# print('.', end='', flush=True)

	# augment images if needed
	if len(samples) < class_size:
		needed = class_size - len(samples)
		# loop for each time that all image augs are needed, then randomly select the remainder
		for _ in range(needed // len(samples)):
			epoch(samples, seq, sizes)
		epoch(samples, seq, sizes, stop=(needed % len(samples)))

def generate_augmentations (mapping, seq, class_size=1000, sizes=(224,)):
	for cid, samples in enumerate(mapping):
		samples = samples.copy()
		random.shuffle(samples)
		process_class_augmentations(samples, class_size, seq, sizes)

# generate test set and return list of files used to remove from dataframes
def process_test_set (mapping, seq, class_size=1000, sizes=(224,)):
	used = []
	needed = class_size
	# if files are already present in test dir don't regenerate them
	for samples in mapping:
		for sample in samples:
			if os.path.isfile(sample[1].replace('[SIZE]', str(sizes[0]))):
				used.append(sample[0].split('/')[-1])
				needed -= 1
	# copy over images
	for cid, samples in enumerate(mapping):
		samples = samples.copy()
		random.shuffle(samples)
		# copy over max(class_size, len(samples)) images
		for sample in samples[:needed]:
			process_image(sample[0], sample[1], sizes)
			used.append(sample[0].split('/')[-1])
	return used

# remove test set from dataframe and return new df
def remove_used (df, used):
	return df.drop(df[
		df['path'].map(lambda x: x.split('/')[-1] in used)
	].index)
