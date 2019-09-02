
import os, sys, cv2
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras.applications.densenet import preprocess_input
sys.path.append('models')
import matplotlib.pyplot as plt

import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)

base_model = keras.models.load_model('models/h5/densenet121-binary.h5')
feature_model = Model(inputs=base_model.input, outputs=base_model.get_layer('batch_normalization').output)
model = feature_model.layers[1]

print('Loaded model. Processing...')

img = (cv2.cvtColor(cv2.imread('images/exemplars/4/1a7e3356b39c.png'), cv2.COLOR_BGR2RGB) / 256).reshape(1, 224, 224, 3)

row = 0 # first row is a little broken
rows = 8
cols = 8
j = -1; nth = 4
fig, axs = plt.subplots(rows, cols, figsize=(10, 10))
plt.axis('off')
for k in range(5, len(model.layers)):
	if "Conv2D" in str(model.layers[k].__class__):
		j += 1
		if j % nth != 0: continue
		act = Model(inputs=model.input, outputs=model.layers[k].output).predict(img)[0]
		# act.shape = (w, h, filters)
		for col in range(cols):
			axs[row][col].imshow(act[:, :, col], cmap='gray')
			axs[row][col].set_xticks([])
			axs[row][col].set_yticks([])
		row += 1
		if row == rows: break

plt.subplots_adjust(
	top=0.99,
	bottom=0.01,
	left=0.01,
	right=0.99,
	hspace=0.05,
	wspace=0.05
)

plt.show()
