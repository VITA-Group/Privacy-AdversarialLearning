import sys, time, os, datetime, errno, pprint, itertools
from os.path import dirname, realpath
sys.path.append(dirname(dirname(dirname(realpath(__file__)))))
sys.path.append(dirname(dirname(dirname(dirname(realpath(__file__))))))

import tensorflow as tf
from common_flags import COMMON_FLAGS
from eval_flags import FLAGS
from modules.degradNet import fd
from modules.targetNet import fT
from loss import *
from utils import *
from nets import nets_factory
from sklearn.metrics import average_precision_score

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ["CUDA_VISIBLE_DEVICES"]=FLAGS.GPU_id

def create_architecture_fT(scope, batch_size, videos, labels, dropout_placeholder):
    videos = tf.reshape(videos, [batch_size * COMMON_FLAGS.DEPTH, COMMON_FLAGS.CROP_HEIGHT, COMMON_FLAGS.CROP_WIDTH, COMMON_FLAGS.NCHANNEL])
    with tf.variable_scope(tf.get_variable_scope(), reuse=False):
        fd_videos = fd(videos)
    fd_videos = tf.reshape(fd_videos, [batch_size, COMMON_FLAGS.DEPTH, COMMON_FLAGS.CROP_HEIGHT, COMMON_FLAGS.CROP_WIDTH, COMMON_FLAGS.NCHANNEL])
    logits_fT = fT(fd_videos, dropout_placeholder)
    loss_fT = tower_loss_xentropy_sparse(logits_fT, labels, use_weight_decay=True, name_scope=scope)
    return loss_fT, logits_fT

def build_graph(gpu_num, batch_size, is_training):
    graph = tf.Graph()
    with graph.as_default():
        videos_placeholder = tf.placeholder(tf.float32, shape=(batch_size * gpu_num, COMMON_FLAGS.DEPTH, None, None, COMMON_FLAGS.NCHANNEL))
        labels_placeholder = tf.placeholder(tf.int64, shape=(batch_size * gpu_num))
        dropout_placeholder = tf.placeholder(tf.float32)
        logits_fT_lst = []
        loss_fT_lst = []
        with tf.variable_scope(tf.get_variable_scope()) as scope:
            for gpu_index in range(0, gpu_num):
                with tf.device('/gpu:%d' % gpu_index):
                    print('/gpu:%d' % gpu_index)
                    with tf.name_scope('%s_%d' % ('gpu', gpu_index)) as scope:
                        videos = videos_placeholder[gpu_index * batch_size:(gpu_index + 1) * batch_size]
                        labels = labels_placeholder[gpu_index * batch_size:(gpu_index + 1) * batch_size]
                        loss_fT, logits_fT = create_architecture_fT(scope, batch_size, videos, labels, dropout_placeholder)
                        logits_fT_lst.append(logits_fT)
                        loss_fT_lst.append(loss_fT)
                        tf.get_variable_scope().reuse_variables()

        logits_fT_op = tf.concat(logits_fT_lst, 0)
        loss_fT_op = tf.reduce_mean(loss_fT_lst)
        right_count_utility_op = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(tf.nn.softmax(logits_fT_op), axis=1),
                                                                labels_placeholder), tf.int32))
        if is_training:
            videos_op, labels_op = create_videos_reading_ops(is_train=True, is_val=False, GPU_NUM=gpu_num, BATCH_SIZE=batch_size, NUM_EPOCHS=1)
        else:
            videos_op, labels_op = create_videos_reading_ops(is_train=False, is_val=False, GPU_NUM=gpu_num, BATCH_SIZE=batch_size, NUM_EPOCHS=1)

        init_op = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())
        return (graph, init_op, videos_placeholder, labels_placeholder, dropout_placeholder,
                loss_fT_op, logits_fT_op, right_count_utility_op, videos_op, labels_op)

def run_testing_fT():
    dir_path = FLAGS.adv_ckpt_dir
    ckpt_files = [".".join(f.split(".")[:-1]) for f in os.listdir(dir_path) if os.path.isfile(
                                        os.path.join(dir_path, f)) and '.data' in f]

    for ckpt_file in ckpt_files:
        for is_training in [True, False]:
            config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
            config.gpu_options.allow_growth = True
            (graph, init_op, videos_placeholder, labels_placeholder, dropout_placeholder,
             loss_fT_op, logits_fT_op, right_count_utility_op, videos_op, labels_op) = build_graph(FLAGS.GPU_NUM, FLAGS.video_batch_size, is_training)
            with tf.Session(graph=graph, config=config) as sess:
                sess.run(init_op)

                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=sess, coord=coord)

                # Create a saver for writing training checkpoints.
                saver = tf.train.Saver(tf.trainable_variables())
                saver.restore(sess, os.path.join(dir_path, ckpt_file))
                print('Session restored from trained model at {}!'.format(os.path.join(dir_path, ckpt_file)))

                total_v_utility = 0.0
                test_correct_num_utility = 0.0
                try:
                    while not coord.should_stop():
                        videos, videos_labels = sess.run([videos_op, labels_op])
                        feed = {videos_placeholder: videos, labels_placeholder: videos_labels, dropout_placeholder: 1.0}
                        right_utility = sess.run(right_count_utility_op, feed_dict=feed)
                        print(total_v_utility)
                        test_correct_num_utility += right_utility
                        total_v_utility += videos_labels.shape[0]

                except tf.errors.OutOfRangeError:
                    print('Done testing on all the examples')
                finally:
                    coord.request_stop()
                coord.join(threads)

                save_dir = os.path.join(FLAGS.adv_ckpt_dir, ckpt_file.split('.')[-1])
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                isTraining = lambda bool: "training" if bool else "testing"
                with open(os.path.join(save_dir, 'fT_accuracy_{}.txt'.format(isTraining(is_training))), 'w') as wf:
                    wf.write('fT test acc: {}\n'.format(test_correct_num_utility / total_v_utility))
                    wf.write('fT test_correct_num: {}\n'.format(test_correct_num_utility))
                    wf.write('fT total_v: {}\n'.format(total_v_utility))

if __name__ == '__main__':
    run_testing_fT()