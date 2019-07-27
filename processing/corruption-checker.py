
from PIL import Image
import glob, os

broken = []

for fname in glob.glob('data/proc/new/224/**/*.png') + glob.glob('data/proc/old/224/**/*.png') + glob.glob('data/proc/aug/224/**/*.png'):
	try:
		Image.open(fname).verify()
	except:
		print(fname)
		broken.append(fname)

if input('delete all? y/n: ') == 'y':
	for fname in broken:
		os.remove(fname)
		os.remove(fname.replace('224','299'))
else:
	print(broken)
