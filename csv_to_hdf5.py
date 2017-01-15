import numpy as np
import sys
import os
import h5py


csv_files = os.listdir('.')
csv_files = [f for f in csv_files if f.isdigit()]

# create hdf5 database
f = h5py.File('step10.hdf5', 'w')

BATCH_SIZE = 64
NUM_FEATURES = 13 * 13 * 115 

dset = f.create_dataset("features",
                        (BATCH_SIZE * len(csv_files), NUM_FEATURES),
                        dtype='float32')

batch = np.empty((BATCH_SIZE, NUM_FEATURES), dtype=np.float32)

for ind, csv in enumerate(csv_files):
    with open(csv) as fcsv:
        lines = fcsv.readlines()
        for i, line in enumerate(lines):
            batch[i] = np.fromstring(line, dtype=np.float32, sep=',')
        dset[BATCH_SIZE * ind:BATCH_SIZE * (ind + 1)] = batch


f.flush()
f.close()
