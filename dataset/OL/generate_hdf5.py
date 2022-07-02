import h5py
import os
from PIL import Image
import numpy as np

def writeItems(outputfile, itemlist):
    with open(outputfile, 'w+') as f:
        for item in itemlist:
            f.write("%s\n" % item)

root = 'dataset'
files = [str(x) for x in np.arange(105)+1]

if not os.path.isdir('hdf5'):
    os.makedirs(os.path.join('hdf5'))

for fname in files:
    polar_list = []
    for deg in [0, 45, 90, 135]:
        img = np.array(Image.open(os.path.join(root, fname, f'gt_{deg}', fname + '.png')).convert('RGB'))/255.
        polar_list.append(np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0))
    gt = np.concatenate(polar_list, axis=0)
    with h5py.File(os.path.join('hdf5', f'{fname}.hdf5'), 'w') as f:
        f.create_dataset("gt", data=gt)
