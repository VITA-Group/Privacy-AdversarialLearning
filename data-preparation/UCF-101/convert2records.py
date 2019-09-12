
# coding: utf-8

# In[4]:


# coding: utf-8

# In[1]:


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from VideoReader import VideoReader
import numpy as np
import cv2

# Not using GPUs
#os.environ["CUDA_VISIBLE_DEVICES"]=""

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convert_to(videos, labels, name, directory):
    if videos.shape[0] != labels.shape[0]:
        raise ValueError('Videos size %d does not match labels size %d.' %
                         (videos.shape[0], labels.shape[0]))
    num_examples = videos.shape[0]

    filename = os.path.join(directory, '{}.tfrecords'.format(name))
    print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)
    for index in range(num_examples):
        video_raw = videos[index].tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'label': _int64_feature(int(labels[index])),
            'video_raw': _bytes_feature(video_raw),
        }))
        writer.write(example.SerializeToString())
    writer.close()

def write_video(videos, labels, width=160, height=120, nchannel=3):
    class_dict = {}
    with open('ucfTrainTestlist/classInd.txt', 'r') as f:
        for line in f:
            print(line)
            words = line.strip('\n').split()
            class_dict[int(words[0]) - 1] = words[1]

    for i in range(len(videos)):
        print(i)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output = '{}.avi'.format(class_dict[labels[i]])
        #output = '{}_{:f}_{:d}.avi'.format(class_dict[labels[i]], sigmas[i], i)
        out = cv2.VideoWriter(output, fourcc, 1.0, (width, height), nchannel==3)
        vid = videos[i]
        vid = vid.astype('uint8')
        print(vid.shape)
        for i in range(vid.shape[0]):
            frame = vid[i]
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame = frame.reshape(height, width, nchannel)
            out.write(frame)
        out.release()
        cv2.destroyAllWindows()

def video_processing(videos_dir):
    depth, height, width = 16, 120, 160
    color = True
    factor = 4
    vreader = VideoReader(depth, height, width, color, factor)
    X_train, Y_train, X_test, Y_test = vreader.loaddata(videos_dir)
    X_train = X_train.reshape((X_train.shape[0], depth, height, width, 3)).astype('uint8')
    X_test = X_test.reshape((X_test.shape[0], depth, height, width, 3)).astype('uint8')
    print('X_train shape:{}\nY_train shape:{}'.
                                    format(X_train.shape, Y_train.shape))
    print('X_test shape:{}\nY_test shape:{}'.
                                    format(X_test.shape, Y_test.shape))
    return X_train, Y_train.astype('uint32'), X_test, Y_test.astype('uint32')

import resource

# In[ ]:
def mem():
    print('Memory usage         : % 2.2f MB' % round(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0,1)
    )

def memoryhog(folder):
    directory = 'data'
    print(os.path.join('UCF-101-shuffled', folder))
    X_train, Y_train, X_test, Y_test = video_processing(videos_dir=os.path.join('UCF-101-shuffled', folder))
    convert_to(X_train, Y_train, 'train_{}'.format(folder), directory)
    convert_to(X_test, Y_test, 'test_{}'.format(folder), directory)
    #write_video(X_train, Y_train)
    #write_video(X_test, Y_test)

if __name__ == "__main__":
    from joblib import Parallel, delayed
    num_cores = 20
    if not os.path.exists('data'):
        os.makedirs('data')
    Parallel(n_jobs=num_cores)(delayed(memoryhog)(q) for q in next(os.walk('UCF-101-shuffled'))[1])
