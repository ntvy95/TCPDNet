import h5py
import os
from PIL import Image
import numpy as np
import scipy.io

root = 'mat'
files = os.listdir(root)

if not os.path.isdir('hdf5'):
    os.makedirs(os.path.join('hdf5'))

rm_end = len('.mat')
for file in files:
    polar_list = []
    data = scipy.io.loadmat(os.path.join(root, file))
    for deg in [0, 45, 90, 135]:
        polar_list.append(np.expand_dims(np.transpose(data[f'RGB_{deg}'], (2, 0, 1)), axis=0))
    gt = np.concatenate(polar_list, axis=0)
    with h5py.File(os.path.join('hdf5', f'{file[:-rm_end]}.hdf5'), 'w') as f:
        f.create_dataset("gt", data=gt)

    
    

