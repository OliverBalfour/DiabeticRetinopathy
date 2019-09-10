
import numpy as np
import os, sys
sys.path.append('./models')
from model_utils import load_xy

X, Y = load_xy('densenet121')

# inverse squared euclidean distance (would manhattan be better?)
def dist (a, b):
	# return np.linalg.norm(a-b)
	s = np.sum((a-b)**2)
	if s == 0:
		# print(a, b)
		return False
	return 1.0/s

def get_nearest_neighbours_densenet (x, i, k=100, limit=-1):

	# choose first k neighbours
	neighbours = [(s, dist(s, x), n) for n, s in enumerate(X[:k])]
	neighbours = sorted(neighbours, key=lambda s: s[1], reverse=True)

	# loop through remaining training data and update neighbours accordingly
	for n, s in enumerate(X[k:limit]):
		if i == n + k: continue

		# inverse square so larger distance is better
		d = dist(s, x)
		if d == False:
			continue
		if d > neighbours[-1][1]:
			# temporarily replace furthest neighbour with current point
			neighbours[-1] = (s, d, n + k) # we start at n=0 for X[k] so n+k is the index

			# now we need to sort it so that the last neighbour is in the correct place
			# everything else is already sorted though
			# thus we choose the first position for which d>neighbours[i][1]
			correct_place = -1
			for i in range(len(neighbours)):
				if d > neighbours[i][1]:
					correct_place = i
					break

			# now we sort
			neighbours = neighbours[:correct_place] + [neighbours[-1]] + neighbours[correct_place:-1]

	indices = [s[2] for s in neighbours]
	inverse_squares = [s[1] for s in neighbours]
	return indices, inverse_squares

# takes a vector in densenet space and returns local accuracy by computing weighted mean of accuracies of nearby points in training data
# also requires Y_hat, the predictions by the model for all of the training data (use all-stacked-outputs?)
def compute_local_accuracy (Y_hat, x, indices, inverse_squares):

	# now that we have the k nearest neighbours we compute the local accuracy using inverse squares as weights
	# we take dot(correctness, inverse_squares)

	# correctness: 0 for incorrect, 1 for correct
	Y_cat = np.argmax(Y, axis=1)
	Y_cat_in_the_hat = np.argmax(Y_hat, axis=1) # y_hat in categorical format. excuse the awful joke
	# print(len(Y_cat))
	# print(len(Y_cat_in_the_hat))
	correctness = 1 - np.abs(Y_cat[indices] - Y_cat_in_the_hat[indices])
	return np.dot(correctness, inverse_squares)
