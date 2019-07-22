
import numpy as np
import pandas as pd
import cv2, os

from process import process, post_process
from dataframes import get_train_df

train_df = get_train_df()

sizes = (224, 299)

# process an image and save it in each specified size
def process_image (read, write):
	img = process(cv2.imread(read))
	for size in sizes:
		cv2.imwrite(
			write.replace('[SIZE]', str(size)),
			post_process(img, size)
		)

# iterate through training data and process it into folders like ./data/proc/224/0/
for row in train_df.itertuples():
	try:
		process_image(row.path, f'data/proc/[SIZE]/{str(row.diagnosis)}/{row.id_code}.png')
		print('.', end='', flush=True)
	except Exception as err:
		print(row.path)
		print(err)

print('Preprocessed training images.')
