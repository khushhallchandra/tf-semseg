import os
import torch
import numpy as np
import scipy.misc as m
from torch.utils import data
from skimage.transform import resize 

class mit_scene_loader(data.Dataset):
    def __init__(self, root, img_size, split="train"):
        
        self.split = split
        self.n_classes = 151
        self.img_size = img_size 
        
        if(self.split=="train"):
            self.root = root
            self.img_dir = self.root + 'images/training/'
            self.lbl_dir = self.root + 'annotations/training/'
            self.img_name = os.listdir(self.img_dir)
            self.lbl_name = os.listdir(self.lbl_dir)
        else:
            self.root = root
            self.img_dir = self.root + 'images/validation/'
            self.lbl_dir = self.root + 'annotations/validation/'
            self.img_name = os.listdir(self.img_dir)
            self.lbl_name = os.listdir(self.lbl_dir)
        
        print("Total number of data sample:",len(self.img_name))

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, index):
        try:
            img = m.imread(self.img_dir + self.img_name[index], mode='RGB')
            msk = m.imread(self.lbl_dir + self.lbl_name[index])

            img = m.imresize(img, self.img_size, 'nearest')  
            msk = m.imresize(msk, self.img_size, 'nearest') 

        except Exception as e:
            print('Failed loading image. Creating dummy data')
            img = np.zeros((self.img_size[0], self.img_size[1], 3))
            msk = np.zeros((self.img_size[0], self.img_size[1]))

        return img, msk
