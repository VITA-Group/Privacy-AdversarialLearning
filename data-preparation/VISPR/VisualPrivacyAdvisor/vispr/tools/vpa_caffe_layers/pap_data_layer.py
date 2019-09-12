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
import scipy

from random import shuffle

from vispr import DS_ROOT, CAFFE_ROOT

sys.path.insert(0, os.path.join(CAFFE_ROOT, 'python'))
import caffe

sys.path.insert(1, CAFFE_ROOT + 'examples/pycaffe/layers')   # the datalayers we will use are in this directory.
sys.path.insert(1, CAFFE_ROOT + 'examples/pycaffe')   # the tools file is in this folder
from tools import SimpleTransformer

__author__ = "Tribhuvanesh Orekondy"
__maintainer__ = "Tribhuvanesh Orekondy"
__email__ = "orekondy@mpi-inf.mpg.de"
__status__ = "Development"


def load_attributes(attr_list_path=None):
    """
    Returns mappings: {attribute_id -> attribute_name} and {attribute_id -> idx}
    where attribute_id = 'aXX_YY' (string),
    attribute_name = description (string),
    idx \in [0, 67] (int)
    :return:
    """
    if attr_list_path is None:
        attributes_path = osp.join(DS_ROOT, 'attributes.tsv')
    else:
        attributes_path = attr_list_path
    attr_id_to_name = dict()
    attr_id_to_idx = dict()

    with open(attributes_path, 'r') as fin:
        ts = csv.DictReader(fin, delimiter='\t')
        rows = filter(lambda r: r['idx'] is not '', [row for row in ts])

        for row in rows:
            attr_id_to_name[row['attribute_id']] = row['description']
            attr_id_to_idx[row['attribute_id']] = int(row['idx'])

    return attr_id_to_name, attr_id_to_idx


def labels_to_vec(labels, attr_id_to_idx):
    n_labels = len(attr_id_to_idx)
    label_vec = np.zeros(n_labels)
    for attr_id in labels:
        label_vec[attr_id_to_idx[attr_id]] = 1
    return label_vec


class PAPMultilabelDataLayer(caffe.Layer):

    """
    This is a simple synchronous datalayer for training a multilabel model on
    PASCAL.
    """

    def setup(self, bottom, top):
        self.top_names = ['data', 'label']

        # params is a python dictionary with layer parameters.
        # Example param_str: param_str:
        # "{\'anno_list\': \'train2017.txt\', \'im_shape\': [227, 227], \'batch_size\': 128}"
        params = eval(self.param_str)

        # Check the parameters for validity.
        check_params(params)

        # Create a batch loader to load the images.
        self.batch_loader = BatchLoader(params, None)
        self.batch_size = params['batch_size']
        self.n_attr = self.batch_loader.n_attr

        # Weighted loss
        self.wloss = bool(params.get('wloss', 0))

        self.pool = params.get('pool', 'max')
        assert self.pool in ['sum', 'max', 'avg']

        # Reshape Top --------------------------------------------------------------------------------------------------
        # since we use a fixed input image size, we can shape the data layer
        # once. Else, we'd have to do it in the reshape call.
        top[0].reshape(
            self.batch_size, 3, params['im_shape'][0], params['im_shape'][1])
        top[1].reshape(self.batch_size, self.n_attr)

        print_info("PAPMultilabelDataLayerSync", params)


    def forward(self, bottom, top):
        """
        Load data.
        """
        # Create a matrix N x A: Labels for each image
        img_attr_mat = np.zeros((self.batch_size, self.n_attr))

        for itt in range(self.batch_size):
            # Use the batch loader to load the next image.
            im, multilabel = self.batch_loader.load_next_image()

            assert multilabel.shape[0] == self.n_attr
            img_attr_mat[itt] = multilabel[:]

            # Add directly to the caffe data layer
            top[0].data[itt, ...] = im
            top[1].data[itt, ...] = multilabel

    def reshape(self, bottom, top):
        """
        There is no need to reshape the data, since the input is of fixed size
        (rows and columns)
        """
        pass

    def backward(self, top, propagate_down, bottom):
        """
        These layers does not back propagate
        """
        pass


class BatchLoader(object):

    """
    This class abstracts away the loading of images.
    Images can either be loaded singly, or in a batch. The latter is used for
    the asyncronous data layer to preload batches while other processing is
    performed.
    """

    def __init__(self, params, result):
        self.result = result
        self.batch_size = params['batch_size']
        self.anno_list = params['anno_list']
        self.im_shape = params['im_shape']
        self.attribute_list_path = params['attribute_list_path']
        self.memimages = bool(params.get('memimages', 0))
        self.img_transform = params.get('img_transform', 'resize')
        self.ynorm = bool(params.get('ynorm', 0))   # y := y / ||y||_1

        # this class does some simple data-manipulations
        self.transformer = SimpleTransformer(mean=[104, 117, 123])
        self.dictlist = []  # Annotation objects (image + labels) will be stored as dict here

        # Load Attributes ----------------------------------------------------------------------------------------------
        self.attr_id_to_name, self.attr_id_to_idx = load_attributes(self.attribute_list_path)
        self.idx_to_attr_id = {v: k for k, v in self.attr_id_to_idx.iteritems()}
        self.attr_id_list = self.attr_id_to_idx.keys()
        self.n_attr = len(self.attr_id_list)

        # Load Data ----------------------------------------------------------------------------------------------------
        # Store the list of annotation files as indexlist
        self.indexlist = [osp.join(DS_ROOT, line.rstrip('\n')) for line in open(self.anno_list)]

        if self.memimages:
            print "Loading images into memory"
        print "Loading {} annotations".format(len(self.indexlist))

        # Store each image-label object as a dict
        # But, do not store the images. Only store the image file path
        self.dictlist = [json.load(open(filename)) for filename in self.indexlist]
        shuffle(self.dictlist)
        self._cur = 0  # current image

        # Add additional information to each dict
        for idx, this_anno in enumerate(self.dictlist):
            # Prepare the multilabel
            # Get the list of attributes this corresponds to
            attr_set = set(this_anno['labels'])
            multilabel = labels_to_vec(attr_set, self.attr_id_to_idx)
            if self.ynorm and np.sum(multilabel) > 0:
                multilabel /= np.sum(multilabel)
            assert np.sum(multilabel) > 0, 'Failed: np.sum(multilabel) > 0'
            this_anno['label_vec'] = multilabel

            this_anno['image_path'] = osp.join(DS_ROOT, this_anno['image_path'])
            # Images can sometimes be huge (>5mb), which makes loading data extremely slow
            # So, resize and stash them to enable quick loading
            image_resized_path = this_anno['image_path'].replace('/images/', '/images_250/')
            if os.path.exists(image_resized_path):
                this_anno['image_path'] = image_resized_path

            # To make training even faster, load the images into memory before it begins
            if self.memimages:
                im = imread(this_anno['image_path'])
                if len(im.shape) == 2:
                    # This is a grayscale image
                    im = np.asarray(Image.open(this_anno['image_path']).convert('RGB'))
                elif len(im.shape) == 3 and im.shape[2] == 4:
                    # CMYK Image
                    im = np.asarray(Image.open(this_anno['image_path']).convert('RGB'))

                if self.img_transform == 'resize':
                    # Resize the image to the required shape
                    im = scipy.misc.imresize(im, self.im_shape)

                this_anno['im'] = im

                if idx % 100 == 0:
                    sys.stdout.write("processing %d/%d (%.2f%% done)   \r" % (
                    idx, len(self.dictlist), idx * 100.0 / len(self.dictlist)))
                    sys.stdout.flush()

        print "BatchLoader initialized with {} images".format(len(self.indexlist))

    def load_next_image(self):
        """
        Load the next image in a batch.
        """

        # Sample image -------------------------------------------------------------------------------------------------
        # Did we finish an epoch?
        if self._cur == len(self.dictlist):
            self._cur = 0
            next_idx = self._cur
            shuffle(self.dictlist)
        else:
            next_idx = self._cur
            self._cur += 1

        dct = self.dictlist[next_idx]  # Get the anno

        # Load image ---------------------------------------------------------------------------------------------------
        image_path = dct['image_path']
        multilabel = dct['label_vec']

        # Load an image
        if 'im' in dct:  # Images can be preloaded before training with flag memimages
            im = dct['im']
        else:
            im = imread(image_path)
            if len(im.shape) == 2:
                # This is a grayscale image
                im = np.asarray(Image.open(image_path).convert('RGB'))
            elif len(im.shape) == 3 and im.shape[2] == 4:
                # CMYK Image
                im = np.asarray(Image.open(image_path).convert('RGB'))
        org_shape = im.shape

        # Resize/Transform image ---------------------------------------------------------------------------------------
        if self.img_transform == 'resize':
            # Resize the image to the required shape
            im = scipy.misc.imresize(im, self.im_shape)
        elif self.img_transform == 'rand_crop':
            # Take a random crop of size self.im_shape
            # im.shape = [H, W, 3]
            img_h, img_w, _ = im.shape
            crop_h, crop_w = self.im_shape

            if img_w < crop_w:
                new_w = crop_w
                new_h = int(np.round(img_h * (new_w / float(img_w))))   # Scale height to same aspect ratio
                im = scipy.misc.imresize(im, (new_h, new_w))
                img_w, img_h = new_w, new_h
                # print 'New (w, h): ', (img_w, img_h)

            if img_h < crop_h:
                new_h = crop_h
                new_w = int(np.round(img_w * (new_h / float(img_h))))
                im = scipy.misc.imresize(im, (new_h, new_w))
                img_w, img_h = new_w, new_h

            # Sample (x1, y1) i.e, top-left point of the image
            x1 = np.random.randint(low=0, high=(img_h - crop_h - 1))
            y1 = np.random.randint(low=0, high=(img_w - crop_w - 1))
            # Crop a window given this point
            x2 = x1 + crop_h
            y2 = y1 + crop_w
            im = im[x1:x2, y1:y2, :]

        # do a simple horizontal flip as data augmentation
        flip = np.random.choice(2)*2-1
        im = im[:, ::flip, :]

        transformed_im = self.transformer.preprocess(im)

        return transformed_im, multilabel

def check_params(params):
    """
    A utility function to check the parameters for the data layers.
    """
    required = ['batch_size', 'anno_list', 'im_shape']
    for r in required:
        assert r in params.keys(), 'Params must include {}'.format(r)


def print_info(name, params):
    """
    Output some info regarding the class
    """
    print "{} initialized for split: {}, with bs: {}, im_shape: {}.".format(
        name,
        params['anno_list'],
        params['batch_size'],
        params['im_shape'])
