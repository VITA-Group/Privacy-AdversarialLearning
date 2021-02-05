#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np 
from skimage.io import imsave
from skvideo.io import vread
import os

read_dir = 'videos'
save_dir = 'frames'

for root, dirs, files in os.walk(read_dir):
    depth = root[len(read_dir):].count(os.path.sep)
    if depth == 1:
        action = root.split('/')[-1]
        print(root)
        for file in files:
            if not file.endswith('.avi'):
                continue
            vid_name = file.split('.')[0]
            path = os.path.join(root, file)
            print(path)
            video = vread(path)
            print('video:', video.shape)
            for i in range(int(video.shape[0])):
                image = video[i,::]
                folder_dir = os.path.join(save_dir, action, vid_name)
                if not os.path.exists(folder_dir):
                    os.makedirs(folder_dir)
                imsave(os.path.join(folder_dir, '{}.png'.format(str(i))), image)

