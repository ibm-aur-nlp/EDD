'''
Prepares data files for training and validating EDD model. The following files
are generated in <out_dir>:
 - TRAIN_IMAGES_<POSTFIX>.h5        # Training images
 - TRAIN_TAGS_<POSTFIX>.json        # Training structural tokens
 - TRAIN_TAGLENS_<POSTFIX>.json     # Length of training structural tokens
 - TRAIN_CELLS_<POSTFIX>.json       # Training cell tokens
 - TRAIN_CELLLENS_<POSTFIX>.json    # Length of training cell tokens
 - VAL.json                         # Validation ground truth

<POSTFIX> is formatted according to input args (keep_AR, max_tag_len, ...)
'''
import json
from tqdm import tqdm
import argparse
import os
import h5py
import numpy as np
from utils import image_rescale


parser = argparse.ArgumentParser(description='Prepares data files for training EDD')
parser.add_argument('--image_dir', type=str, help='path to image folder')
parser.add_argument('--out_dir', type=str, help='path to folder to save data files')
parser.add_argument('--image_size', default=448, type=int, help='target image rescaling size')

args = parser.parse_args()

with open(os.path.join(args.out_dir, 'VAL.json'), 'r') as fp:
    VAL = json.load(fp)
val_image_paths = [os.path.join(args.image_dir, k) for k, _ in VAL.items()]

if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)

with h5py.File(os.path.join(args.out_dir, 'VAL_IMAGES.hdf5'), 'a') as h:
    # Create dataset inside HDF5 file to store images
    images = h.create_dataset('images', (len(val_image_paths), 3, args.image_size, args.image_size), dtype='uint8')

    for i, path in enumerate(tqdm(val_image_paths)):
        # Read images
        img = image_rescale(path, args.image_size, False)
        assert img.shape == (3, args.image_size, args.image_size)
        assert np.max(img) <= 255

        # Save image to HDF5 file
        images[i] = img
