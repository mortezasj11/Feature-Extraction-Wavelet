# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 09:05:25 2021

@author: MBSaad
"""

# Imports
import pathlib
import torch
from skimage.io import imread
from torch.utils import data
from tqdm.notebook import tqdm




import albumentations
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from skimage.transform import resize
#from customdatasets import SegmentationDataSet1, SegmentationDataSet2
from niftiReader import SegmentationDataSet2
from transformations import ComposeDouble, AlbuSeg2d, FunctionWrapperDouble, normalize_01, create_dense_target
from unet_recon_ver2 import UNet
from trainer import Trainer



# data directory
root = pathlib.Path.cwd() / 'Lung_public'



def get_filenames_of_path(path: pathlib.Path, ext: str = '*'):
    """Returns a list of files in a directory/path. Uses pathlib."""
    filenames = [file for file in path.glob(ext) if file.is_file()]
    return filenames

# input and target files
inputs = get_filenames_of_path(root / 'Input')
targets = get_filenames_of_path(root / 'Target')

#pre-transformation (resize input image to 128 128)

pre_transforms = ComposeDouble([
    FunctionWrapperDouble(resize,
                          input=True,
                          target=False,
                          output_shape=(128, 128, 1)),
                          #output_shape=(128, 128)),
    FunctionWrapperDouble(resize,
                          input=False,
                          target=True,
                          # output_shape=(1,128, 128),
                          output_shape=(128,128,1),
                          # order=0,
                          # anti_aliasing=False,
                          # preserve_range=True
                          ),
])

# training transformations and augmentations
transforms_training = ComposeDouble([
    AlbuSeg2d(albumentations.HorizontalFlip(p=0.5)),
    # FunctionWrapperDouble(create_dense_target, input=False, target=True),
    FunctionWrapperDouble(np.moveaxis, input=True, target=True, source=-1, destination=0),
    FunctionWrapperDouble(normalize_01)
])


# validation transformations
transforms_validation = ComposeDouble([
    # FunctionWrapperDouble(create_dense_target, input=False, target=True),
    FunctionWrapperDouble(np.moveaxis, input=True, target=True, source=-1, destination=0),
    FunctionWrapperDouble(normalize_01)
])


# random seed
random_seed = 42

# split dataset into training set and validation set
train_size = 0.8  # 80:20 split


inputs_train, inputs_valid = train_test_split(
     inputs,
     random_state=random_seed,
     train_size=train_size,
     shuffle=True)

targets_train, targets_valid = train_test_split(
     targets,
     random_state=random_seed,
     train_size=train_size,
     shuffle=True)

#inputs_train, inputs_valid = inputs[:200], inputs[200:220]
#targets_train, targets_valid = targets[:200], targets[200:220]

# dataset training
dataset_train = SegmentationDataSet2(inputs=inputs_train,
                                    targets=targets_train,
                                    transform=transforms_training,
                                    use_cache=True,
                                    pre_transform=pre_transforms)

# dataset validation
dataset_valid = SegmentationDataSet2(inputs=inputs_valid,
                                    targets=targets_valid,
                                    transform=transforms_validation,
                                    use_cache=True,
                                    pre_transform=pre_transforms)

# dataloader training
dataloader_training = DataLoader(dataset=dataset_train,
                                 batch_size=2,
                                 shuffle=True)

# dataloader validation
dataloader_validation = DataLoader(dataset=dataset_valid,
                                   batch_size=2,
                                   shuffle=True)

