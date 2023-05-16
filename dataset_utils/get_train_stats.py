"""
This file is used to calculate the mean
and standard deviation for each color
channel in a given training set. These
statistics are used elsewhere to normalize
the input data to the various models.
The statistics for the CelebA 1 corr
and 2 corr training datasets have
already been computed, and you can
find them in settings.py.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from settings import IMG_HEIGHT, IMG_WIDTH, TRAIN_DIR

print('RUNNING')

NUM_CLASSES = 2
BATCH_SIZE = 512
img_size = (IMG_WIDTH, IMG_HEIGHT)
assert img_size == (75, 75), "Images must be 75x75"
AREA = 75*75
data_transforms = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor()
])

train_loader = DataLoader(datasets.ImageFolder(TRAIN_DIR, transform=data_transforms),
                        batch_size=BATCH_SIZE, shuffle=False)

# calculate the means
red_sum, green_sum,blue_sum, rgb_sum, pixel_count = 0,0,0,0,0
for i, (images, _) in enumerate(train_loader):
    red_sum += torch.sum(images[:,0]).item()
    green_sum += torch.sum(images[:,1]).item()
    blue_sum += torch.sum(images[:,2]).item()
    pixel_count += (AREA * images.size(0))

rgb_sum += (red_sum + green_sum + blue_sum)

red_mean = red_sum / pixel_count
green_mean = green_sum / pixel_count
blue_mean = blue_sum / pixel_count

print('Means: ')
print('red_mean: ', red_mean)
print('green_mean: ', green_mean)
print('blue_mean: ', blue_mean)

# calculate the stdevs
red_res, green_res, blue_res = 0,0,0
for i, (images, _) in enumerate(train_loader):
    red_res += torch.sum(torch.square(images[:,0] - red_mean)).item()
    green_res += torch.sum(torch.square(images[:,1] - green_mean)).item()
    blue_res += torch.sum(torch.square(images[:,2] - blue_mean)).item()

red_stdev = np.sqrt(red_res / pixel_count)
green_stdev = np.sqrt(green_res / pixel_count)
blue_stdev = np.sqrt(blue_res / pixel_count)

print('stdevs: ')
print('red_stdev: ', red_stdev)
print('green_stdev: ', green_stdev)
print('blue_stdev: ', blue_stdev)
