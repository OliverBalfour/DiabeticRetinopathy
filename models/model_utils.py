

# alphanumerically sorts all files in data/proc/224 and returns the diagnoses for each
def get_sorted_classes ():
	train_dir = f'data/proc/224/'
	files = []
	for cid in range(5):
		for x in os.listdir(train_dir + str(cid)):
			files.append(x+str(cid))

	files.sort()
	classes = [int(x[-1]) for x in files]
	return classes

# alphanumerically sorts all files in data/proc/224 and returns their their dataset
# True is new, old is False
def get_sorted_datasources ():
	train_dir = f'data/proc/224/'
	files = []
	for cid in range(5):
		for x in os.listdir(train_dir + str(cid)):
			files.append(x+str(cid))

	files.sort()
	datasources = [1 - (x.endswith('left.png') or x.endswith('right.png')) for x in files]
	return datasources
