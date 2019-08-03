
import numpy as np
import pandas as pd
import cv2, os

from image_functions import process_image
from dataframes import get_train_df_new, get_train_df_old

train_df_new = get_train_df_new()
train_df_old = get_train_df_old()

sizes = (224,) # 299 as well?

# iterate through training data and process it into folders like ./data/proc/new/224/0/
def iterate_df (df, dirname):
	for row in df.itertuples():
		try:
			process_image(row.path, f'{dirname}[SIZE]/{str(row.diagnosis)}/{row.id_code}.png', sizes)
			print('.', end='', flush=True)
		except Exception as err:
			print(row.path)
			print(err)

iterate_df(train_df_new, 'data/proc/new/')
iterate_df(train_df_old, 'data/proc/old/')

print('\nPreprocessed training images.')
