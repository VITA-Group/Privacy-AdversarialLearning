'''
Author: Haotao
Convert val set and testing set from tfrecords to ndarray, so that we can val on the whole validation set.
'''
import sys, os
import tensorflow as tf
import numpy as np

from utils import create_videos_reading_ops
from common_flags import COMMON_FLAGS

GPU_id = "1"
GPU_NUM = int((len(GPU_id)+1)/2)

BATCH_SIZE = 8

os.environ["CUDA_VISIBLE_DEVICES"]=GPU_id

def tfrecord2ndarray(mode):
    '''
    Conver tfrecord to ndarray
    Args:
        mode: string. val or test.
    '''
    assert(mode in ['val', 'test'])

    graph = tf.Graph()
    with graph.as_default():
        videos_op, action_labels_op, actor_labels_op = create_videos_reading_ops(
            is_train=False, is_val=(mode in ['val']), GPU_NUM=GPU_NUM, BATCH_SIZE=BATCH_SIZE)

    # run session:
    with tf.Session(graph=graph) as sess:
        # initialization:
        init_op = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())
        sess.run(init_op)
        # initialization part should be put outside the multi-threads part! But why?

        # multi-threads:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        
        total_v = 0.0 # total number of testing samples

        print('coord.should_stop():', coord.should_stop())
        try:
            c = 0
            test_videos_lst, test_action_labels_lst, test_actor_labels_lst = [], [], []
            while not coord.should_stop():
                c += 1
                print('in while loop ', str(c))
                # input operations:
                videos, action_labels, actor_labels = sess.run([videos_op, action_labels_op, actor_labels_op])
                total_v += action_labels.shape[0]
                print('total_v:', total_v)
                print('videos:', videos.shape)
                print('action_labels:', action_labels.shape)
                print('actor_labels:', actor_labels.shape)
                test_videos_lst.append(videos)
                test_action_labels_lst.append(action_labels)
                test_actor_labels_lst.append(actor_labels)

        except tf.errors.OutOfRangeError:
            videos = np.concatenate(test_videos_lst, axis=0)
            action_labels = np.concatenate(test_action_labels_lst, axis=0)
            actor_labels = np.concatenate(test_actor_labels_lst, axis=0)
            print('videos:', videos.shape)
            print('action_labels:', action_labels.shape)
            print('actor_labels:', actor_labels.shape)
            np.save(os.path.join(COMMON_FLAGS.FILES_NPY_DIR, '%s_videos.npy') % mode, videos)
            np.save(os.path.join(COMMON_FLAGS.FILES_NPY_DIR, '%s_action_labels.npy') % mode, action_labels)
            np.save(os.path.join(COMMON_FLAGS.FILES_NPY_DIR, '%s_actor_labels.npy') % mode, actor_labels)
            print('Done converting all %s examples' % mode)
        finally:
            coord.request_stop()
        
        coord.join(threads)
        sess.close()

if __name__ == '__main__':
    tfrecord2ndarray('val')
    print('=====================================\n')
    tfrecord2ndarray('test')
