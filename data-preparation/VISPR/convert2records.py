#!/usr/bin/python

import json
import sys
from skimage.io import imread
from utils import *
import tensorflow as tf
import os
import os.path as osp
from PIL import Image

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def convert_to(images, labels, name, directory):
    print(images.shape)
    print(labels.shape)
    if images.shape[0] != labels.shape[0]:
        raise ValueError('Images size %d does not match labels size %d.' %
                         (images.shape[0], labels.shape[0]))
    num_examples = images.shape[0]
    height = images.shape[1]
    width = images.shape[2]
    nchannel = images.shape[3]

    filename = os.path.join(directory, '{}.tfrecords'.format(name))
    print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)
    for index in range(num_examples):
        image_raw = images[index].astype(np.float32).tostring()
        label_raw = labels[index].astype(np.float32).tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(height),
            'width': _int64_feature(width),
            'nchannel': _int64_feature(nchannel),
            'label_raw': _bytes_feature(label_raw),
            'image_raw': _bytes_feature(image_raw)
        }))
        writer.write(example.SerializeToString())
    writer.close()

def main():
    anno_file_list = 'train2017.txt'
    # Load Annotations -------------------------------------------------------------------------------------------------
    attr_id_to_name, attr_id_to_idx = load_attributes('../PA_HMDB51/attributes_pa_hmdb51.csv')
    print(attr_id_to_name)
    print(attr_id_to_idx)
    anno_path_list = []  # Store anno paths
    anno_list = []   # Store anno dicts
    with open(anno_file_list) as f:
        for line in f:
            anno_path = line.strip()
            with open(anno_path) as jf:
                anno_path_list.append(anno_path)
                anno_list.append(json.load(jf))
    print('Found {} annotation files'.format(len(anno_list)))

    n_files = len(anno_list)

    images_lst = []
    labels_lst = []
    # Perform Forward-Passes per image ---------------------------------------------------------------------------------
    for idx in range(0, len(anno_list)):
        sys.stdout.write("Processing %d/%d (%.2f%% done) \r" % (idx + 1, n_files,
                                                                (idx + 1) * 100.0 / n_files))
        sys.stdout.flush()
        anno = anno_list[idx]
        image_path = anno['image_path']
        attr_set = set(anno['labels'])
        attr_vec = labels_to_vec(attr_set, attr_id_to_idx)
        print(attr_vec.shape)
        image_resized_path = image_path.replace('images', 'images_112')
        try:
            image = imread(image_resized_path)
        except ValueError:
            print('Unable to process: ', image_resized_path)
            print(image.shape)
        if len(image.shape) == 2:
            # This is a grayscale image
            image = np.asarray(Image.open(image_resized_path).convert('RGB'))
        elif len(image.shape) == 3 and image.shape[2] == 4:
            # CMYK Image
            image = np.asarray(Image.open(image_resized_path).convert('RGB'))
        images_lst.append(image)
        labels_lst.append(attr_vec)
        print(image.shape)
    print(np.asarray(labels_lst).astype('uint8').shape)
    print(np.asarray(images_lst).astype('uint8').shape)
    convert_to(np.asarray(images_lst).astype('uint8'), np.asarray(labels_lst).astype('uint8'), name='vispr_train_112', directory='.')

if __name__ == '__main__':
    main()
