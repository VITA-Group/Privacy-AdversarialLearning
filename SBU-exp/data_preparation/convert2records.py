
# coding: utf-8

# In[2]:

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import cv2
from SBU.VideoReader import SBUReader
import resource
#os.environ["CUDA_VISIBLE_DEVICES"]=""

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convert_to(videos, action_labels, actor_labels, name, directory):
    if videos.shape[0] != action_labels.shape[0]:
        raise ValueError('Videos size %d does not match action labels size %d.' %
                         (videos.shape[0], action_labels.shape[0]))
    if videos.shape[0] != actor_labels.shape[0]:
        raise ValueError('Videos size %d does not match actor labels size %d.' %
                         (videos.shape[0], actor_labels.shape[0]))

    num_examples = videos.shape[0]

    filename = os.path.join(directory, name + '.tfrecords')
    print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)
    for index in range(num_examples):
        video_raw = videos[index].tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'action_label': _int64_feature(int(action_labels[index])),
            'actor_label': _int64_feature(int(actor_labels[index])),
            'video_raw': _bytes_feature(video_raw)
        }))
        writer.write(example.SerializeToString())
    writer.close()

def video_processing(videos_dir):
    vreader = SBUReader(depth=16, sigma=1.0, ksize=4)
    #X, Y_action, Y_actor = vreader.loaddata_LR(videos_dir, train=False)

    train_lst, val_lst, test_lst = vreader.loaddata_HR(videos_dir)
    X, Y_action, Y_actor = train_lst[0], train_lst[1], train_lst[2]

    print('X shape:{}\nY_action shape:{}\nY_actor shape:{}'.format(X.shape,
                                                                    Y_action.shape, Y_actor.shape))

    return train_lst, val_lst, test_lst


def mem():
    print('Memory usage: % 2.2f MB' % round(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0,1)
    )

def memoryhog(folder):
    path = os.path.join('/home/wuzhenyu_sjtu/Data_Preparation/SBU_videos/noisy_version/RGB', folder)
    print(path)
    train_lst, val_lst, test_lst = video_processing(path)
    convert_to_records(train_lst, 'training_noisy_{}'.format(folder))
    convert_to_records(val_lst, 'validation_noisy_{}'.format(folder))
    convert_to_records(test_lst, 'testing_noisy_{}'.format(folder))


def convert_to_records(lst, name):
    directory = '/home/wuzhenyu_sjtu/Data_Preparation/tfrecords'
    X, Y_action, Y_actor = lst[0], lst[1], lst[2]
    convert_to(X, Y_action, Y_actor, name, directory)

if __name__ == "__main__":
    from joblib import Parallel, delayed
    num_cores = 5
    Parallel(n_jobs=num_cores)(delayed(memoryhog)(q) for q in next(os.walk('/home/wuzhenyu_sjtu/Data_Preparation/SBU_videos/noisy_version/RGB'))[1])
