"""
Dataset class for loading images and their labels.
"""


import pickle as plk
from PIL import Image
import numpy as np
import os
import torch


class Dataset():
    def __init__(self, data_type, transform=None):

        # Loading data path(name)
        with open(data_type + '_data.plk', 'rb') as fin:
            self.ids = plk.load(fin)
            
        # Loading the corresponding labels
        with open(data_type + '_lbl.plk', 'rb') as fin:
            self.lbls = plk.load(fin)

        self.transform = transform

    def __getitem__(self, index):
        data_name = self.ids[index]
        lbl = self.lbls[index]

	# Loading the image
        img0 = Image.open(data_name)
        #print(img0.size)
        
        if self.transform is not None:
            img1 = self.transform(img0)

        dic = {'pixel_values':img1, 'label':lbl}
        
        return dic

    def __len__(self):
        return len(self.ids)
