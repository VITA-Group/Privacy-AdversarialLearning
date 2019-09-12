
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
from KTH.VideoReader import VideoReader
import resource
import scipy

os.environ["CUDA_VISIBLE_DEVICES"]=""

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

def convert_to_SR(videos_LR, videos_HR, name, directory):
    if videos_LR.shape[0] != videos_HR.shape[0]:
        raise ValueError('Videos LR size %d does not match Videos HR size %d.' %
                         (videos_LR.shape[0], videos_HR.shape[0]))

    num_examples = videos_LR.shape[0]

    filename = os.path.join(directory, name + '.tfrecords')
    print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)
    for index in range(num_examples):
        video_raw_lr = videos_LR[index].tostring()
        video_raw_hr = videos_HR[index].tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'video_raw_hr': _bytes_feature(video_raw_hr),
            'video_raw_lr': _bytes_feature(video_raw_lr)
        }))
        writer.write(example.SerializeToString())
    writer.close()

def write_images(images, labels, num_batch, batch_size):
    actor_dict = {
        0:'person01',
        1:'person02',
        2:'person03',
        3:'person04',
        4:'person05',
        5:'person06',
        6:'person07',
        7:'person08',
        8:'person09',
        9:'person10',
        10:'person11',
        11:'person12',
        12:'person13',
        13:'person14',
        14:'person15',
        15:'person16',
        16:'person17',
        17:'person18',
        18:'person19',
        19:'person20',
        20:'person21',
        21:'person22',
        22:'person23',
        23:'person24',
        24:'person25',
        }
    for i in range(len(images)):
        scipy.misc.imsave('{}_{}.png'.format(actor_dict[labels[i]], i + num_batch * batch_size), images[i])


def write_video(X, Y_action, Y_actor):
    action_dict = {
            0:'running',
            1:'walking',
            2:'jogging',
            3:'handwaving',
            4:'handclapping',
            5:'boxing'
        }
    actor_dict = {
        0:'person01',
        1:'person02',
        2:'person03',
        3:'person04',
        4:'person05',
        5:'person06',
        6:'person07',
        7:'person08',
        8:'person09',
        9:'person10',
        10:'person11',
        11:'person12',
        12:'person13',
        13:'person14',
        14:'person15',
        15:'person16',
        16:'person17',
        17:'person18',
        18:'person19',
        19:'person20',
        20:'person21',
        21:'person22',
        22:'person23',
        23:'person24',
        24:'person25',
        }
    width, height = 160, 120
    for i in range(len(X)):
        fourcc = cv2.VideoWriter_fourcc(*'XVID') # Be sure to use lower case
        if Y_action is None or Y_actor is None:
            output = "{}_{}_{}.avi".format('person25', 'boxing', i)
        else:
            output = "{}_{}_{}.avi".format(actor_dict[Y_actor[i]], action_dict[Y_action[i]], i)
        out = cv2.VideoWriter(output, fourcc, 30.0, (width, height), True)
        vid = X[i]
        vid = vid.astype('uint8')
        print(vid.shape)
        for i in range(vid.shape[0]):
            frame = vid[i]
            frame = frame.reshape(120, 160, 3)
            out.write(frame)
    out.release()
    cv2.destroyAllWindows()

def video_processing(videos_dir):
    depth, height, width = 16, 120, 160
    nchannel = 3
    vreader = VideoReader(depth, height, width, sigma=1.0, ksize=4)
    #X, Y_action, Y_actor = vreader.loaddata_SR(videos_dir)
    #X = X.reshape((X.shape[0], depth, height, width, nchannel)).astype('uint8')
    #Y_action = Y_action.astype('uint32')
    #Y_actor = Y_actor.astype('uint32')
    #print('X_shape:{}\nY_action_shape:{}\nY_actor_shape:{}'.format(X.shape, Y_action.shape, Y_actor.shape))
    #return X, Y_action, Y_actor

    X_LR, X_HR = vreader.loaddata_SR(videos_dir)
    X_LR = X_LR.reshape((X_LR.shape[0], depth, height, width, nchannel)).astype('uint8')
    X_HR = X_HR.reshape((X_HR.shape[0], depth, height, width, nchannel)).astype('uint8')
    return X_LR, X_HR


def mem():
    print('Memory usage         : % 2.2f MB' % round(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0,1)
    )

def memoryhog(folder):
    directory = '/hdd/wuzhenyu_sjtu/'
    path = os.path.join('test', folder)
    print(path)
    #X, Y_action, Y_actor = video_processing(path)
    #convert_to(X, Y_action, Y_actor, 'training_{}'.format(folder), directory)
    #write_video(X, Y_action, Y_actor)
    X_LR, X_HR = video_processing(path)
    convert_to_SR(X_LR, X_HR, 'testing{}'.format(folder), directory)
    #write_video(X_HR, None, None)

if __name__ == "__main__":
    from joblib import Parallel, delayed
    num_cores = 15
    Parallel(n_jobs=num_cores)(delayed(memoryhog)(q) for q in next(os.walk('test'))[1])
