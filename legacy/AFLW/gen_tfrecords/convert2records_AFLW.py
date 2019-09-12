from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from torchvision import transforms
import torch
from gen_tfrecords.AFLW import AFLW

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def convert_to(images, labels_yaw, labels_pitch, labels_roll,
               labels_yaw_cont, labels_pitch_cont, labels_roll_cont, genders,
               name, directory):
    if images.shape[0] != labels_yaw.shape[0]:
        raise ValueError('Images size %d does not match labels size %d.' %
                         (images.shape[0], labels_yaw.shape[0]))
    num_examples = images.shape[0]
    height = images.shape[1]
    width = images.shape[2]
    nchannel = images.shape[3]

    filename = os.path.join(directory, '{}.tfrecords'.format(name))
    print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)
    for index in range(num_examples):
        image_raw = images[index].astype(np.float32).tostring()
        label_yaw = labels_yaw[index].astype(np.float32)
        label_pitch = labels_pitch[index].astype(np.float32)
        label_roll = labels_roll[index].astype(np.float32)
        label_yaw_cont = labels_yaw_cont[index].astype(np.float32)
        label_pitch_cont = labels_pitch_cont[index].astype(np.float32)
        label_roll_cont = labels_roll_cont[index].astype(np.float32)
        gender = genders[index].astype(np.int32)
        #identity = identities[index].astype(np.int32)
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(height),
            'width': _int64_feature(width),
            'nchannel': _int64_feature(nchannel),
            'label_yaw_raw': _floats_feature(label_yaw),
            'label_pitch_raw': _floats_feature(label_pitch),
            'label_roll_raw': _floats_feature(label_roll),
            'label_yaw_cont_raw': _floats_feature(label_yaw_cont),
            'label_pitch_cont_raw': _floats_feature(label_pitch_cont),
            'label_roll_cont_raw': _floats_feature(label_roll_cont),
            'gender': _int64_feature(gender),
            'image_raw': _bytes_feature(image_raw)
        }))
        writer.write(example.SerializeToString())
    writer.close()

if __name__ == "__main__":
    #transformations = transforms.Compose([transforms.Scale(240),
    #transforms.RandomCrop(224), transforms.ToTensor(),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    transformations = transforms.Compose([transforms.Scale(240), transforms.CenterCrop(224), transforms.ToTensor()])

    pose_dataset = AFLW('/home/wuzhenyu_sjtu/deep-head-pose',
                        '/home/wuzhenyu_sjtu/deep-head-pose/AFLW/train_list', transformations)

    train_loader = torch.utils.data.DataLoader(dataset=pose_dataset, batch_size=1, shuffle=True,
                                               drop_last=False, num_workers=1)

    arr_image, arr_label_yaw, arr_label_pitch, arr_label_roll = [], [], [], []
    arr_label_yaw_cont, arr_label_pitch_cont, arr_label_roll_cont = [], [], []
    arr_gender = []
    for i, (images, labels, cont_labels, gender) in enumerate(train_loader):
        images = np.transpose(images.cpu().numpy(), [0, 2, 3, 1])
        labels = labels.cpu().numpy()
        cont_labels = cont_labels.cpu().numpy()
        gender = gender.cpu().numpy()

        # Binned labels
        label_yaw = labels[:, 0][0]
        label_pitch = labels[:, 1][0]
        label_roll = labels[:, 2][0]

        # Continuous labels
        label_yaw_cont = cont_labels[:, 0][0]
        label_pitch_cont = cont_labels[:, 1][0]
        label_roll_cont = cont_labels[:, 2][0]
        #print(label_yaw, label_pitch, label_roll)
        #print(label_yaw_cont, label_pitch_cont, label_roll_cont)
        #print(label_yaw.shape, label_pitch.shape, label_roll.shape)
        #print(label_yaw_cont.shape, label_pitch_cont.shape, label_roll_cont.shape)
        #print(images)
        print('Sample [%d/%d] ' % (i + 1, len(pose_dataset)))

        arr_image.append(np.squeeze(images))
        arr_label_yaw.append(np.asscalar(label_yaw))
        arr_label_pitch.append(np.asscalar(label_pitch))
        arr_label_roll.append(np.asscalar(label_roll))
        arr_label_yaw_cont.append(np.asscalar(label_yaw_cont))
        arr_label_pitch_cont.append(np.asscalar(label_pitch_cont))
        arr_label_roll_cont.append(np.asscalar(label_roll_cont))
        arr_gender.append(np.asscalar(gender[:, 0]))
        #arr_identity.append(np.asscalar(identity[:, 0]))
        #print(arr_gender)
        #print(arr_identity)
        if (i+1) % 1000 == 0:
            convert_to(np.asarray(arr_image), np.asarray(arr_label_yaw), np.asarray(arr_label_pitch), np.asarray(arr_label_roll),
                       np.asarray(arr_label_yaw_cont), np.asarray(arr_label_pitch_cont), np.asarray(arr_label_roll_cont),
                       np.asarray(arr_gender), name='training_{}'.format(i), directory='tfrecords')
            arr_image, arr_label_yaw, arr_label_pitch, arr_label_roll = [], [], [], []
            arr_label_yaw_cont, arr_label_pitch_cont, arr_label_roll_cont = [], [], []
            arr_gender, arr_identity = [], []
    convert_to(np.asarray(arr_image), np.asarray(arr_label_yaw), np.asarray(arr_label_pitch), np.asarray(arr_label_roll),
               np.asarray(arr_label_yaw_cont), np.asarray(arr_label_pitch_cont), np.asarray(arr_label_roll_cont),
               np.asarray(arr_gender), name='training_{}'.format(len(pose_dataset)), directory='tfrecords')