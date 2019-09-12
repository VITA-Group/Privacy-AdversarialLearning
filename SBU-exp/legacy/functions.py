#!/usr/bin/env python

import sys
sys.path.insert(0, '..')

import time
from six.moves import xrange
import input_data
import errno
import pprint
import itertools
from degradlNet import residualNet
from budgetNet import budgetNet
from utilityNet import utilityNet
from loss import *
from utils import *
from img_proc import _avg_replicate
import yaml
from tf_flags import FLAGS

try:
  xrange
except:
  xrange = range

def placeholder_inputs(batch_size, cfg):
    videos_placeholder = tf.placeholder(tf.float32, shape=(batch_size, cfg['DATA']['DEPTH'], 112, 112, cfg['DATA']['NCHANNEL']))
    labels_placeholder = tf.placeholder(tf.int64, shape=(batch_size))
    istraining_placeholder = tf.placeholder(tf.bool)
    return videos_placeholder, labels_placeholder, istraining_placeholder

def create_videos_reading_ops(is_train, is_val, cfg):
    '''
    Multi-thread data fetching from queue
    '''
    train_files = [os.path.join(FLAGS.train_files_dir, f) for f in
                    os.listdir(FLAGS.train_files_dir) if f.endswith('.tfrecords')]
    val_files = [os.path.join(FLAGS.val_files_dir, f) for f in
                    os.listdir(FLAGS.val_files_dir) if f.endswith('.tfrecords')]
    test_files = [os.path.join(FLAGS.test_files_dir, f) for f in
                    os.listdir(FLAGS.test_files_dir) if f.endswith('.tfrecords')]

    num_threads = cfg['DATA']['NUM_THREADS']
    num_examples_per_epoch = cfg['TRAIN']['NUM_EXAMPLES_PER_EPOCH']
    if is_train:
        batch_size = cfg['TRAIN']['BATCH_SIZE'] * FLAGS.GPU_NUM
        videos_op, action_labels_op, actor_labels_op = input_data.inputs_videos(filenames=train_files,
                                                                                batch_size=batch_size,
                                                                                num_epochs=None,
                                                                                num_threads=num_threads,
                                                                                num_examples_per_epoch=num_examples_per_epoch,
                                                                                shuffle=True)
    elif is_val:
        batch_size = cfg['TRAIN']['BATCH_SIZE'] * FLAGS.GPU_NUM
        videos_op, action_labels_op, actor_labels_op = input_data.inputs_videos(filenames=val_files,
                                                                                batch_size=batch_size,
                                                                                num_epochs=None,
                                                                                num_threads=num_threads,
                                                                                num_examples_per_epoch=num_examples_per_epoch,
                                                                                shuffle=True)
    else:
        batch_size = cfg['TRAIN']['BATCH_SIZE'] * FLAGS.GPU_NUM # considering testing and training shares the same graph, we should use same batch_size here.
        videos_op, action_labels_op, actor_labels_op = input_data.inputs_videos(filenames=test_files,
                                                                                batch_size=batch_size,
                                                                                num_epochs=1,
                                                                                num_threads=num_threads,
                                                                                num_examples_per_epoch=num_examples_per_epoch,
                                                                                shuffle=False)

    return videos_op, action_labels_op, actor_labels_op