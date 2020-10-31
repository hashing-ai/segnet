#!/usr/bin/env python3

# Learning how to take data and prepare it for Neural Training

# ----- imports ----- #

# pytorch imports
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# data manipulation and visualization imports
import cv2
import numpy as np
import matplotlib.pyplot as plt

# misc imports
import os
import glob

plt.ion()

# ------------------- #

class SegDataset(Dataset):

    def __init__(self, imageFolder, maskFolder, transform=None, fraction=None, seed=None, subset=None):
        self.imageFolder = imageFolder
        self.maskFolder = maskFolder
        self.transform = transform

        # if fraction = None
        if not fraction:
            self.image_paths = sorted(glob.glob(os.path.join(self.imageFolder,"*")))
            self.mask_paths = sorted(glob.glob(os.path.join(self.maskFolder,"*")))

        # if fraction value is given
        else:
            self.fraction = fraction
            self.image_array = np.array(sorted(glob.glob(os.path.join(self.imageFolder,"*"))))
            self.mask_array = np.array(sorted(glob.glob(os.path.join(self.maskFolder,"*"))))

            if seed:
                np.random.seed(seed)
                indices = np.arange(len(self.image_array))
                np.random.shuffle(indices)  # changes indices array and randomizes it

                self.image_array = self.image_array[indices]
                self.mask_array = self.mask_array[indices]

            if subset == 'train':
                self.image_paths = self.image_array[:int(np.ceil(len(self.image_array)*(self.fraction)))]
                self.mask_paths = self.mask_array[:int(np.ceil(len(self.mask_array)*(self.fraction)))]

            if subset == 'test':
                self.image_paths = self.image_array[int(np.ceil(len(self.image_array)*(self.fraction))):]
                self.mask_paths = self.mask_array[int(np.ceil(len(self.mask_array)*(self.fraction))):]



    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_name = self.image_paths[idx]
        msk_name = self.mask_paths[idx]

        # note : cv2 and pytorch handle images differently

        image = cv2.imread(img_name, 1).transpose(2,0,1)
        mask = cv2.imread(msk_name, 0)

        sample = {'image': image, 'mask': mask}

        if self.transform:
            sample = self.transform(sample)

        return sample


# Creating callable classes for each transform

class Rescale(object):
    '''Rescale the image in a sample to a given size

    Args:
        output_size (tuple) : Desired output size
    '''

    def __init__(self, img_resize, msk_resize):
        self.img_resize = img_resize
        self.msk_resize = msk_resize

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        if len(image.shape) == 3:
            image = image.transpose(1,2,0)
        if len(mask.shape) == 3:
            mask = mask.transpose(1,2,0)

        image = cv2.resize(image, self.img_resize, cv2.INTER_AREA)
        mask = cv2.resize(mask, self.msk_resize, cv2.INTER_AREA)

        if len(image.shape) == 3:
            image = image.transpose(2,0,1)
        if len(mask.shape) == 3:
            mask = mask.transpose(2,0,1)

        return {'image': image, 'mask': mask}


class ToTensor(object):
    def __call__(self, sample, img_resize=None, msk_resize=None):
        image, mask = sample['image'], sample['mask']

        if len(mask.shape) == 2:
            mask = mask.reshape((1,)+mask.shape)

        return {'image': torch.from_numpy(image),
                'mask': torch.from_numpy(mask)}

class Normalize(object):
    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']

        return {'image': image.type(torch.FloatTensor)/255,
                'mask': mask.type(torch.FloatTensor)/255}


# making the data loader #
def make_dataloader(img_dir, msk_dir, seed=44, fraction=0.8, batch_size=16):
    # define transform
    data_transforms = {'train': transforms.Compose([ToTensor(), Normalize()]),
                       'test': transforms.Compose([ToTensor(), Normalize()])}
    # getting datasets
    datasets = {x: SegDataset(img_dir, msk_dir, seed=seed, fraction=fraction, subset=x, transform=data_transforms[x]) for x in ['train', 'test']}
    # getting dataloaders
    dataloaders = {x: DataLoader(datasets[x], batch_size=batch_size, shuffle=True) for x in ['train', 'test']}

    return dataloaders


