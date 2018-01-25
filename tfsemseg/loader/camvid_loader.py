import os
import cv2
import collections
import torch
import torchvision
import numpy as np
import scipy.misc as m
import matplotlib.pyplot as plt

from torch.utils import data

class camvidLoader(data.Dataset):
    def __init__(self, root, split="train", img_size=None):
        if(split=="train"):
            self.root = root + 'train'
        elif(split == "test"):
            self.root = root + 'test'
        else:
            self.root = root + 'val'
        self.split = split
        self.img_size = img_size
        self.n_classes = 12
        self.img_name = os.listdir(self.root)
        self.lbl_name = os.listdir(self.root + 'annot')


    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, index):

        img = m.imread(self.root + '/' + self.img_name[index])
        img = np.array(self.normalized(img), dtype=np.uint8)
        msk = m.imread(self.root + 'annot/'+self.lbl_name[index])
        msk = msk.astype(int)
        return img, msk

    def normalized(self, rgb):
        norm=np.zeros((rgb.shape[0], rgb.shape[1], 3),np.float32)
        b=rgb[:,:,0]
        g=rgb[:,:,1]
        r=rgb[:,:,2]
        norm[:,:,0]=cv2.equalizeHist(b)
        norm[:,:,1]=cv2.equalizeHist(g)
        norm[:,:,2]=cv2.equalizeHist(r)

        return norm
