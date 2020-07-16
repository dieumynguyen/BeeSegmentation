import os
import glob
import numpy as np

# Pytorch
import torch
from torch.utils.data import Dataset

class BeeDataset(Dataset):
    def __init__(self, data_root, transforms=None):
        self.data_root = data_root
        self.transforms = transforms

        self.set_paths()

    def set_paths(self):
        print('Loading paths...')
        self.image_paths = np.sort(glob.glob(f'{self.data_root}/images/*.npy'))
        self.mask_paths = np.array([p.replace('/images/', '/masks/') for p in self.image_paths])

        # Check for existence
        valid_path_idxs = np.array([os.path.exists(p) for p in self.mask_paths])

        if not np.all(valid_path_idxs):
            frac_exist = sum(valid_path_idxs) / len(valid_path_idxs) * 100
            print(f'Fraction of existing image/mask pairs: {frac_exist}')

            # Ignore missing pair
            self.image_paths = self.image_paths[valid_path_idxs]
            self.mask_paths = self.mask_paths[valid_path_idxs]

        self.n_paths = len(self.image_paths)
        print(f'Num paths loaded: {self.n_paths}')

    def __len__(self):
        return self.n_paths

    def __getitem__(self, idx):
        image = np.load(self.image_paths[idx])
        mask = np.load(self.mask_paths[idx])
        if self.transforms:
            image, mask = self.transforms((image, mask))

        return image, mask
