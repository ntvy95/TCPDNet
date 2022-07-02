# A modification of https://github.com/pytorch/vision/blob/master/references/segmentation/transforms.py

import numpy as np
from PIL import Image
import random

import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, gt):
        for t in self.transforms:
            gt = t(gt)
        return gt        

class Random90Rotation(object):        
    def __call__(self, gt):
        degree = random.choice([0, 90, 180, 270])
        return rotation(degree, gt)

def rotation(degree, gt):
    if degree == 0:
        return gt
    elif degree == 90:
        gt = gt.transpose(-1,-2).flip(-2)
    elif degree == 180:
        gt = gt.flip(-2).flip(-1)
    elif degree == 270:
        gt = gt.transpose(-1,-2).flip(-1)
    if degree == 90 or degree == 270:
        gt = gt.index_copy(0, torch.tensor([2,3,0,1]), gt)
    return gt


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, gt):
        crop_params = T.RandomCrop.get_params(gt, (self.size, self.size))
        gt = F.crop(gt, *crop_params)
        return gt
