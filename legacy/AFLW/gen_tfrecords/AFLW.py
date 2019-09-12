import os
import numpy as np
import cv2
import pandas as pd

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms

from PIL import Image, ImageFilter

import utils

class AFLW(Dataset):
    def __init__(self, data_dir, filename_path, transform, img_ext='.jpg', annot_ext='.txt', image_mode='RGB'):
        self.data_dir = data_dir
        self.transform = transform
        self.img_ext = img_ext
        self.annot_ext = annot_ext

        filename_list = self.get_list_from_filenames(filename_path)

        self.X_train = filename_list
        self.y_train = filename_list
        self.image_mode = image_mode
        self.length = len(filename_list)
        self.gender_symbol = lambda s: 0 if s == 'f' else 1

    def get_list_from_filenames(self, file_path):
        # input:    relative path to .txt file with file names
        # output:   list of relative path names
        with open(file_path) as f:
            lines = f.read().splitlines()
        return lines

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.data_dir, self.X_train[index] + self.img_ext))
        img = img.convert(self.image_mode)
        txt_path = os.path.join(self.data_dir, self.y_train[index] + self.annot_ext)

        # We get the pose in radians
        annot = open(txt_path, 'r')
        line = annot.readline().split('\t')
        print(line)
        pose = [float(line[1]), float(line[2]), float(line[3])]
        # And convert to degrees.
        yaw = pose[0] * 180 / np.pi
        pitch = pose[1] * 180 / np.pi
        roll = pose[2] * 180 / np.pi
        # Fix the roll in AFLW
        roll *= -1
        # Bin values
        bins = np.array(range(-99, 102, 3))
        labels = torch.LongTensor(np.clip(np.digitize([yaw, pitch, roll], bins), 1, 66) - 1)
        cont_labels = torch.FloatTensor([yaw, pitch, roll])
        gender = line[4]
        gender = torch.IntTensor([self.gender_symbol(gender)])

        if self.transform is not None:
            img = self.transform(img)

        return img, labels, cont_labels, gender

    def __len__(self):
        # train: 18,863
        # test: 1,966
        return self.length