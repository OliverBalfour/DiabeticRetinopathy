import os

def mvall (fnames, source, dest):
	for fname in fnames:
		os.rename(source+fname,dest+fname)

def det_old (source):
	old = []
	for fname in os.listdir(source):
		if fname.endswith('_right.png') or fname.endswith('_left.png'):
			old.append(fname)
	return old

for size in [224,299]:
	for cid in range(5):
		source='data/proc/'+str(size)+'/'+str(cid)+'/'
		mvall(det_old(source),source,'data/proc/old/'+str(size)+'/'+str(cid)+'/')
