
import numpy as np
import pandas as pd
import cv2, os
import matplotlib.pyplot as plt

from dataframes import get_train_df

train_df = get_train_df()

no_samples = 3
class_names = ['No Retinopathy', 'Mild Nonproliferative', 'Moderate Nonproliferative', 'Severe Nonproliferative', 'Proliferative Retinopathy']
classes = np.arange(len(class_names))

# list of sample id coes by diagnosis in order 012340123... (not 000111... so that plotting is easier)
samples = np.ndarray.flatten(np.array([
	train_df.loc[train_df.diagnosis == class_id].sample(no_samples).id_code.tolist() for class_id in classes
]).T)

def draw_samples (images):
	fig = plt.figure(figsize=(15, 8))
	for i, id_code in enumerate(samples):
		ax = fig.add_subplot(no_samples, len(classes), i+1, xticks=[], yticks=[])
		plt.imshow(images[i]/255)
		if i // len(classes) == 0:
			ax.set_title(class_names[i%len(classes)])
	plt.show()

from process import process, post_process

# process an image and save it in each specified size
def process_image (read, size):
	return post_process(process(cv2.imread(read)), size)

draw_samples([
	cv2.imread('data/proc/299/'+str(int(train_df.loc[train_df.id_code==id_code].sample().diagnosis))+'/'+id_code+'.png')
	# process_image(train_df.loc[train_df.id_code==id_code].path.item(), 299)
	for id_code in samples
])
