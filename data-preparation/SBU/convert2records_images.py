
# coding: utf-8

# In[8]:


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from skimage.io import imread
from matlab_imresize import imresize


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


# In[2]:
def modcrop(image, scale=4):
    """
    To scale down and up the original image, first thing to do is to have no remainder while scaling operation.

    We need to find modulo of height (and width) and scale factor.
    Then, subtract the modulo from height (and width) of original image size.
    There would be no remainder even after scaling operation.
    """
    if len(image.shape) == 3:
        h, w, _ = image.shape
        h = h - np.mod(h, scale)
        w = w - np.mod(w, scale)
        image = image[0:h, 0:w, :]
    else:
        h, w = image.shape
        h = h - np.mod(h, scale)
        w = w - np.mod(w, scale)
        image = image[0:h, 0:w]
    return image

def preprocess_py(path, scale=4):
    """
    Preprocess single image file
    (1) Read original image as YCbCr format (and grayscale as default)
    (2) Normalize
    (3) Apply image file with bicubic interpolation

    Args:
    path: file path of desired file
    input_: image applied bicubic interpolation (low-resolution)
    label_: image with original resolution (high-resolution)
    """

    im_uint8 = imread(path)
    im_double = im_uint8 / 255.

    input_ = modcrop(im_double, scale)
    label_ = imresize(input_, scalar_scale=1./scale)

    return label_

def convert_to_images(images, labels, name, directory):
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

def prepare_data(dataset):
    if dataset == "train":
        data_dir = os.path.join('/home/wuzhenyu_sjtu/DAN/SBU/SBU_frames', dataset)
        data = glob.glob(os.path.join(data_dir, "*.png"))
    else:
        data_dir = os.path.join('/home/wuzhenyu_sjtu/DAN/SBU/SBU_frames', dataset)
        data = glob.glob(os.path.join(data_dir, "*.png"))
        print(data)

    return data

def get_subimages(data, dataset):
    sub_input_sequence = []
    sub_label_sequence = []
    pbar = tqdm(total=len(data), )
    image_size = 33
    label_size = 33
    #padding = abs(image_size - label_size) // 2
    padding = 0
    for i in range(len(data)):
        pbar.update(1)
        label_ = preprocess_py(data[i], 4)
        input_ = label_
        if len(input_.shape) == 3:
            h, w, _ = input_.shape
        else:
            h, w = input_.shape
        for x in range(0, h-image_size+1, 14):
            for y in range(0, w-image_size+1, 14):
                sub_input = input_[x:x+image_size, y:y+image_size] # [33 x 33]
                sub_label = label_[x+padding:x+padding+label_size, y+padding:y+padding+label_size] # [27 x 27]

                # Make channel value
                sub_input = sub_input.reshape([image_size, image_size, 3])
                sub_label = sub_label.reshape([label_size, label_size, 3])

                sub_input_sequence.append(sub_input)
                sub_label_sequence.append(sub_label)
    pbar.close()

    """
    len(sub_input_sequence) : the number of sub_input (33 x 33 x ch) in one image
    (sub_input_sequence[0]).shape : (33, 33, 1)
    """
    # Make list to numpy array. With this transform
    arrdata = np.asarray(sub_input_sequence) # [?, 33, 33, 1]
    arrlabel = np.asarray(sub_label_sequence) # [?, 21, 21, 1]
    print(arrdata.shape)
    print(arrlabel.shape)
    return arrdata, arrlabel


def get_images(data):
    input_sequence = []
    label_sequence = []
    pbar = tqdm(total=len(data), )
    for i in range(len(data)):
        pbar.update(1)
        label_ = preprocess_py(data[i], 2)
        input_ = label_
        if len(input_.shape) == 3:
            h, w, _ = input_.shape
        else:
            h, w = input_.shape
        input_sequence.append(input_)
        label_sequence.append(label_)
    pbar.close()

    """
    len(sub_input_sequence) : the number of sub_input (33 x 33 x ch) in one image
    (sub_input_sequence[0]).shape : (33, 33, 1)
    """
    # Make list to numpy array. With this transform
    arrdata = np.asarray(input_sequence) # [?, 33, 33, 1]
    arrlabel = np.asarray(label_sequence) # [?, 21, 21, 1]
    print(arrdata.shape)
    print(arrlabel.shape)
    return arrdata, arrlabel

# In[12]:

if __name__ == "__main__":
    import random
    filenames = []
    for path, subdirs, files in os.walk('/home/wuzhenyu_sjtu/vpa/pap_real_data/UCF-101_frames'):
        print(path)
        if len(files) > 0:
            filenames.append(os.path.join(path, random.choice(files)))
        #for name in files:
        #    filenames.append(os.path.join(path, name))
    from sklearn.model_selection import train_test_split
    x_train, x_test = train_test_split(filenames, test_size=0.1)
    savepath = '/home/wuzhenyu_sjtu/Data_Preparation'
    arrdata, arrlabel = get_images(x_train)
    convert_to_images(arrdata, arrlabel, name='train', directory=savepath)
    arrdata, arrlabel = get_images(x_test)
    convert_to_images(arrdata, arrlabel, name='val', directory=savepath)
    #print(filenames)
    '''
    savepath = os.path.join('/home/wuzhenyu_sjtu/DAN', 'checkpoint')
    for dataset in ["test", "train"]:
        data = prepare_data(dataset)
        #arrdata, arrlabel = get_subimages(data, dataset)
        arrdata, arrlabel = get_images(data, dataset)
        convert_to_images(arrdata, arrlabel, name=dataset, directory=savepath)
    '''