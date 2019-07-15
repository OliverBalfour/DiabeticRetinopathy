
import numpy as np
import pandas as pd
import cv2, os

from process import preprocess
from dataframes import get_train_df

train_df = get_train_df()

for row in train_df.itertuples():
	try:
		preprocess(row.path, row.id_code, f'data/proc/{str(row.diagnosis)}/')
	except:
		print(row.path)

print('Preprocessed training images.')
