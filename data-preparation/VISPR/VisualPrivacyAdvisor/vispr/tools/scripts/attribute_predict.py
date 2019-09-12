#!/usr/bin/python
"""This is a short description.

Replace this with a more detailed description of what this file contains.
"""
import json
import time
import pickle
import sys
import csv
import argparse
import os
import os.path as osp
import shutil

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from scipy.misc import imread

from vispr import DS_ROOT, CAFFE_ROOT
from vispr.tools.common.utils import *

sys.path.insert(0, os.path.join(CAFFE_ROOT, 'python'))
import caffe

sys.path.insert(1, CAFFE_ROOT + 'examples/pycaffe/layers')   # the datalayers we will use are in this directory.
sys.path.insert(1, CAFFE_ROOT + 'examples/pycaffe')   # the tools file is in this folder

__author__ = "Tribhuvanesh Orekondy"
__maintainer__ = "Tribhuvanesh Orekondy"
__email__ = "orekondy@mpi-inf.mpg.de"
__status__ = "Development"


def classify_paths(net, transformer, anno_file_list, out_file, batch_size=64):
    # Load Annotations -------------------------------------------------------------------------------------------------
    anno_path_list = []  # Store anno paths
    anno_list = []   # Store anno dicts
    with open(anno_file_list) as f:
        for line in f:
            anno_path = osp.join(DS_ROOT, line.strip())
            with open(anno_path) as jf:
                anno_path_list.append(anno_path)
                anno_list.append(json.load(jf))
    print 'Found {} annotation files'.format(len(anno_list))

    n_files = len(anno_list)

    net_img_size = net.blobs['data'].data.shape[-1]
    assert net_img_size in [224, 227, 321]

    # Perform Forward-Passes per image ---------------------------------------------------------------------------------
    for start_idx in range(0, len(anno_list), batch_size):
        sys.stdout.write("Processing %d/%d (%.2f%% done) \r" % (start_idx + 1, n_files,
                                                                (start_idx + 1) * 100.0 / n_files))
        sys.stdout.flush()

        end_idx = min(start_idx + batch_size, len(anno_list))
        this_batch_size = end_idx - start_idx
        this_batch = np.zeros((this_batch_size, 3, net_img_size, net_img_size))

        net.blobs['data'].reshape(this_batch_size,  # batch size
                                  3,  # 3-channel (BGR) images
                                  net_img_size, net_img_size)

        for idx, f_idx in enumerate(range(start_idx, end_idx)):
            image_path = osp.join(DS_ROOT, anno_list[f_idx]['image_path'])
            image_resized_path = image_path.replace('/images/', '/images_250/')
            if os.path.exists(image_resized_path):
                image_path = image_resized_path
            try:
                image = caffe.io.load_image(image_path)
                transformed_image = transformer.preprocess('data', image)
            except ValueError:
                print 'Unable to process: ', image_path
                print image.shape
            this_batch[idx] = transformed_image

        net.blobs['data'].data[...] = this_batch

        output = net.forward()
        output_probs = output['prob']

        output_scores = output['fc9'] if 'fc9' in output else None

        with open(out_file, 'a') as wf:
            for idx, f_idx in enumerate(range(start_idx, end_idx)):
                anno_path = anno_path_list[f_idx]
                dct_entry = {'anno_path': anno_path, 'pred_probs': output_probs[idx].tolist()}
                if output_scores is not None:
                    dct_entry['priv_scores'] = output_scores[idx].tolist()
                json_str = json.dumps(dct_entry)
                wf.write(json_str + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("deploy", type=str, help="Path to caffe deploy proto file")
    parser.add_argument("weights", type=str, help="Path to .caffemodel")
    parser.add_argument("infile", type=str, help="List of annotation paths to classify")
    parser.add_argument("outfile", type=str, help="Prefix for path to write the classification output to")
    parser.add_argument("-d", "--device", type=int, choices=[0, 1, 2, 3], default=0, help="GPU device id")
    parser.add_argument("-b", "--batch_size", type=int, choices=range(512), default=64, help="Batch size")
    args = parser.parse_args()

    params = vars(args)

    # Initialize Network -----------------------------------------------------------------------------------------------
    caffe.set_device(params['device'])
    caffe.set_mode_gpu()

    model_def = params['deploy']
    model_weights = params['weights']

    net = caffe.Net(model_def,  # defines the structure of the model
                    model_weights,  # contains the trained weights
                    caffe.TEST)  # use test mode (e.g., don't perform dropout)

    # Set up transformer -----------------------------------------------------------------------------------------------
    # load the mean ImageNet image (as distributed with Caffe) for subtraction
    mu = np.load(osp.join(CAFFE_ROOT, 'python/caffe/imagenet/ilsvrc_2012_mean.npy'))
    mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
    # mu = np.asarray([111.0, 102.0, 116.0])
    # mu = np.asarray([104.0, 117.0, 123.0])
    # print 'mean-subtracted values:', zip('BGR', mu)

    # create transformer for the input called 'data'
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

    transformer.set_transpose('data', (2, 0, 1))  # move image channels to outermost dimension
    transformer.set_mean('data', mu)  # subtract the dataset-mean value in each channel
    transformer.set_raw_scale('data', 255)  # rescale from [0, 1] to [0, 255]
    transformer.set_channel_swap('data', (2, 1, 0))  # swap channels from RGB to BGR

    # Classify ---------------------------------------------------------------------------------------------------------
    classify_paths(net, transformer, params['infile'], params['outfile'], batch_size=params['batch_size'])


if __name__ == '__main__':
    main()
