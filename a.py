import numpy as np
import os
train_dir = 'data/proc/224/'
num = [len(os.listdir(train_dir + str(cid))) for cid in range(5)]

print('Ran Acc: ' + str(max(num)/np.sum(num)))

