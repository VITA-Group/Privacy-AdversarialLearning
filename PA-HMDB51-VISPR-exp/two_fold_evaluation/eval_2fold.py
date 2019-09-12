from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))
sys.path.append(dirname(dirname(dirname(realpath(__file__)))))

import errno
import itertools
import pprint
import time

import numpy as np
import os
import tensorflow as tf

from input_data import *
from nets import nets_factory

from eval_flags import FLAGS
from sklearn.metrics import average_precision_score
from data_preparation.VISPR.utils import *
from modules.degradNet import fd
from common_flags import COMMON_FLAGS
from utils import *

slim = tf.contrib.slim
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


def placeholder_inputs(batch_size):
    images_placeholder = tf.placeholder(tf.float32, shape=(batch_size, COMMON_FLAGS.CROP_HEIGHT, COMMON_FLAGS.CROP_WIDTH, COMMON_FLAGS.NCHANNEL))
    labels_placeholder = tf.placeholder(tf.float32, shape=(batch_size, COMMON_FLAGS.NUM_CLASSES_BUDGET))
    isTraining_placeholder = tf.placeholder(tf.bool)
    return images_placeholder, labels_placeholder, isTraining_placeholder

def tower_loss_xentropy_sparse(name_scope, logits, labels):
    labels = tf.cast(labels, tf.int64)
    cross_entropy_mean = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    return cross_entropy_mean

def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

def get_varlists():
    varlist_fb = [v for v in tf.trainable_variables() if
                      any(x in v.name for x in ["InceptionV1", "InceptionV2",
                                                "resnet_v1_50", "resnet_v1_101", "resnet_v2_50", "resnet_v2_101",
                                                "MobilenetV1_1.0", "MobilenetV1_0.75", "MobilenetV1_0.5",
                                                'MobilenetV1_0.25'])]
    varlist_fd = [v for v in tf.trainable_variables() if v not in varlist_fb]
    return varlist_fb, varlist_fd

def create_architecture(scope, fb, images, labels):
    fd_images = fd(images)
    logits, _ = fb(fd_images)
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
    return loss, logits

def build_graph(gpu_num, batch_size, fb_name):
    graph = tf.Graph()
    with graph.as_default():
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        images_placeholder, labels_placeholder, isTraining_placeholder = placeholder_inputs(batch_size * gpu_num)
        tower_grads = []
        logits_lst = []
        loss_lst = []
        opt = tf.train.AdamOptimizer(1e-4)
        fb = nets_factory.get_network_fn(fb_name,
                                         num_classes=COMMON_FLAGS.NUM_CLASSES_BUDGET,
                                         weight_decay=FLAGS.weight_decay,
                                         is_training=isTraining_placeholder)
        with tf.variable_scope(tf.get_variable_scope()) as scope:
            for gpu_index in range(0, gpu_num):
                with tf.device('/gpu:%d' % gpu_index):
                    print('/gpu:%d' % gpu_index)
                    with tf.name_scope('%s_%d' % ('gpu', gpu_index)) as scope:
                        images = images_placeholder[gpu_index * batch_size:(gpu_index + 1) * batch_size]
                        labels = labels_placeholder[gpu_index * batch_size:(gpu_index + 1) * batch_size]
                        loss, logits = create_architecture(scope, fb, images, labels)
                        logits_lst.append(logits)
                        loss_lst.append(loss)
                        print([v.name for v in tf.trainable_variables()])
                        varlist_fb, varlist_fd = get_varlists()
                        tower_grads.append(opt.compute_gradients(loss, varlist_fb))
                        tf.get_variable_scope().reuse_variables()
        loss_op = tf.reduce_mean(loss_lst)
        logits_op = tf.concat(logits_lst, 0)

        zero_ops, accum_ops, apply_gradient_op = create_grad_accum_for_late_update(opt,
                                    tower_grads, varlist_fb, FLAGS.n_minibatches, global_step, update_ops_depend=True, decay_with_global_step=False)
        tr_images_op, tr_labels_op = create_images_reading_ops(is_train=True, is_val=False,
                                                                      GPU_NUM=gpu_num, BATCH_SIZE=batch_size)
        val_images_op, val_labels_op = create_images_reading_ops(is_train=False, is_val=True,
                                                                    GPU_NUM=gpu_num, BATCH_SIZE=batch_size)
        init_op = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())
        varlist_bn = [g for g in tf.global_variables() if 'moving_mean' in g.name]
        varlist_bn += [g for g in tf.global_variables() if 'moving_variance' in g.name]

        return (graph, init_op,
                zero_ops, accum_ops, apply_gradient_op,
                loss_op, logits_op,
                tr_images_op, tr_labels_op,
                val_images_op, val_labels_op,
                images_placeholder, labels_placeholder, isTraining_placeholder,
                varlist_fb, varlist_fd, varlist_bn)

def update_fb(sess, step, n_minibatches, zero_fb_op, apply_gradient_fb_op, accum_fb_op, loss_fb_op,
              images_op, images_labels_op, images_placeholder, fb_images_labels_placeholder,
              isTraining_placeholder):
    start_time = time.time()
    sess.run(zero_fb_op)
    loss_fb_lst = []
    for _ in itertools.repeat(None, n_minibatches):
        images, images_labels = sess.run([images_op, images_labels_op])
        _, loss_fb = sess.run([accum_fb_op, loss_fb_op],
                                  feed_dict={images_placeholder: images,
                                             fb_images_labels_placeholder: images_labels,
                                             isTraining_placeholder: True})
        loss_fb_lst.append(loss_fb)
    sess.run([apply_gradient_fb_op])
    assert not np.isnan(np.mean(loss_fb_lst)), 'Model diverged with loss = NaN'
    loss_summary = 'Step: {:4d}, time: {:.4f}, fb loss: {:.8f}'.format(
        step,
        time.time() - start_time, np.mean(loss_fb_lst))
    return loss_summary

def eval_fb(sess, step, n_minibatches, logits_fb_op, loss_fb_op, images_op, images_labels_op,
            images_placeholder, fb_images_labels_placeholder, isTraining_placeholder):
    start_time = time.time()
    loss_fb_lst = []
    pred_probs_lst = []
    gt_lst = []
    for _ in itertools.repeat(None, n_minibatches):
        images, images_labels = sess.run([images_op, images_labels_op])
        gt_lst.append(images_labels)
        logits_fb, loss_fb = sess.run([logits_fb_op, loss_fb_op],
                                              feed_dict={images_placeholder: images,
                                                         fb_images_labels_placeholder: images_labels,
                                                         isTraining_placeholder: True})
        loss_fb_lst.append(loss_fb)
        pred_probs_lst.append(logits_fb)

    pred_probs_mat = np.concatenate(pred_probs_lst, axis=0)
    gt_mat = np.concatenate(gt_lst, axis=0)
    n_examples, n_labels = gt_mat.shape
    print('# Examples = ', n_examples)
    print('# Labels = ', n_labels)
    eval_summary = "Step: {:4d}, time: {:.4f}, Macro MAP = {:.2f}".format(
                    step,
                    time.time() - start_time,
                    100 * average_precision_score(gt_mat, pred_probs_mat, average='macro'))
    return eval_summary


def run_training(fd_ckpt_file, ckpt_dir, fb_name, max_steps, train_from_scratch, ckpt_path):
    batch_size = 128
    # Create model directory
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    (graph, init_op,
     zero_fb_op, accum_fb_op, apply_gradient_fb_op,
     loss_fb_op, logits_fb_op,
     tr_images_op, tr_labels_op,
     val_images_op, val_labels_op,
     images_placeholder, labels_placeholder, isTraining_placeholder,
     varlist_fb, varlist_fd, varlist_bn) = build_graph(FLAGS.GPU_NUM, FLAGS.batch_size, fb_name)
    continue_from_trained_model = False

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True

    with tf.Session(graph=graph, config=config) as sess:
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        if continue_from_trained_model:
            saver = tf.train.Saver(varlist_fb+varlist_fd+varlist_bn)
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir=ckpt_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Session restored from pretrained fb + fd at {}!'.format(ckpt.model_checkpoint_path))
            else:
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), ckpt_dir)
        else:
            saver = tf.train.Saver(varlist_fd)
            print(fd_ckpt_file)
            saver.restore(sess, fd_ckpt_file)
            if not train_from_scratch:
                varlist = [v for v in varlist_fb+varlist_bn if not any(x in v.name for x in ["logits"])]
                restore_from_model_zoo_ckpt(sess, ckpt_path, varlist, fb_name)

        saver = tf.train.Saver(varlist_fb+varlist_fd+varlist_bn, max_to_keep=1)
        for step in range(max_steps):
            loss_summary = update_fb(sess, step, FLAGS.n_minibatches, zero_fb_op, apply_gradient_fb_op, accum_fb_op, loss_fb_op,
                      tr_images_op, tr_labels_op, images_placeholder, labels_placeholder, isTraining_placeholder)
            print(loss_summary)
            if step % FLAGS.val_step == 0:
                eval_summary = eval_fb(sess, step, 30, logits_fb_op, loss_fb_op, tr_images_op, tr_labels_op,
                        images_placeholder, labels_placeholder, isTraining_placeholder)
                print("TRAINING: "+eval_summary)
                eval_summary = eval_fb(sess, step, 30, logits_fb_op, loss_fb_op, val_images_op, val_labels_op,
                        images_placeholder, labels_placeholder, isTraining_placeholder)
                print("VALIDATION: "+eval_summary)

            # Save a checkpoint and evaluate the model periodically.
            if step % FLAGS.save_step == 0 or (step + 1) == max_steps:
                checkpoint_path = os.path.join(ckpt_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

        coord.request_stop()
        coord.join(threads)


def run_testing(ckpt_dir, fb_name, is_training):
    batch_size = 128
    # Create model directory
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    (graph, init_op,
     zero_fb_op, accum_fb_op, apply_gradient_fb_op,
     loss_fb_op, logits_fb_op,
     tr_images_op, tr_labels_op,
     val_images_op, val_labels_op,
     images_placeholder, labels_placeholder, isTraining_placeholder,
     varlist_fb, varlist_fd, varlist_bn) = build_graph(FLAGS.GPU_NUM, FLAGS.batch_size, fb_name)

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True

    with tf.Session(graph=graph, config=config) as sess:
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        saver = tf.train.Saver(varlist_fb+varlist_fd+varlist_bn)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir=ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Session restored from trained fb + fd model at {}!'.format(ckpt.model_checkpoint_path))
        else:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), ckpt_dir)

        loss_budget_lst = []
        pred_probs_lst = []
        gt_lst = []
        try:
            while not coord.should_stop():
                images, labels = sess.run([val_images_op, val_labels_op])
                gt_lst.append(labels)
                feed = {images_placeholder: images, labels_placeholder: labels,
                        isTraining_placeholder: False}
                logits, loss = sess.run([logits_fb_op, loss_fb_op], feed_dict=feed)
                loss_budget_lst.append(loss)
                pred_probs_lst.append(logits)
        except tf.errors.OutOfRangeError:
            print('Done testing on all the examples')
        finally:
            coord.request_stop()
        coord.join(threads)
        sess.close()

        pred_probs_mat = np.concatenate(pred_probs_lst, axis=0)
        gt_mat = np.concatenate(gt_lst, axis=0)
        n_examples, n_labels = gt_mat.shape
        isTraining = lambda bool: "training" if bool else "validation"
        with open(os.path.join(ckpt_dir, '{}_{}_class_scores.txt'.format(fb_name, isTraining(is_training))), 'w') as wf:
            wf.write('# Examples = {}\n'.format(n_examples))
            wf.write('# Labels = {}\n'.format(n_labels))
            wf.write('Macro MAP = {:.2f}\n'.format(100 * average_precision_score(gt_mat, pred_probs_mat, average='macro')))
            cmap_stats = average_precision_score(gt_mat, pred_probs_mat, average=None)
            attr_id_to_name, attr_id_to_idx = load_attributes('../../data_preparation/PA_HMDB51/attributes_pa_hmdb51.csv')
            idx_to_attr_id = {v: k for k, v in attr_id_to_idx.items()}
            wf.write('\t'.join(['attribute_id', 'attribute_name', 'num_occurrences', 'ap']) + '\n')
            for idx in range(n_labels):
                attr_id = idx_to_attr_id[idx]
                attr_name = attr_id_to_name[attr_id]
                attr_occurrences = np.sum(gt_mat, axis=0)[idx]
                ap = cmap_stats[idx]
                wf.write('{}\t{}\t{}\t{}\n'.format(attr_id, attr_name, attr_occurrences, ap * 100.0))

def main(_):
    pp = pprint.PrettyPrinter()
    pp.pprint(FLAGS)
    ckpt_base = os.path.join(COMMON_FLAGS.hdd_dir, 'model_zoo')

    ckpt_path_map = {
        'inception_v1': os.path.join(ckpt_base, 'inception_v1/inception_v1.ckpt'),
        'inception_v2': os.path.join(ckpt_base, 'inception_v2/inception_v2.ckpt'),
        'resnet_v1_50': os.path.join(ckpt_base, 'resnet_v1_50/resnet_v1_50.ckpt'),
        'resnet_v1_101': os.path.join(ckpt_base, 'resnet_v1_101/resnet_v1_101.ckpt'),
        'resnet_v2_50': os.path.join(ckpt_base, 'resnet_v2_50/resnet_v2_50.ckpt'),
        'resnet_v2_101': os.path.join(ckpt_base, 'resnet_v2_101/resnet_v2_101.ckpt'),
        'mobilenet_v1': os.path.join(ckpt_base, 'mobilenet_v1_1.0_128/'),
        'mobilenet_v1_075': os.path.join(ckpt_base, 'mobilenet_v1_0.75_128/'),
        'mobilenet_v1_050': os.path.join(ckpt_base, 'mobilenet_v1_0.50_128/'),
        'mobilenet_v1_025': os.path.join(ckpt_base, 'mobilenet_v1_0.25_128/'),
    }
    model_max_steps_map = {
        'inception_v1': 400,
        'inception_v2': 400,
        'resnet_v1_50': 400,
        'resnet_v1_101': 400,
        'resnet_v2_50': 400,
        'resnet_v2_101': 400,
        'mobilenet_v1': 400,
        'mobilenet_v1_075': 400,
        'mobilenet_v1_050': 1000,
        'mobilenet_v1_025': 1000,
    }
    model_train_from_scratch_map = {
        'inception_v1': False,
        'inception_v2': False,
        'resnet_v1_50': False,
        'resnet_v1_101': False,
        'resnet_v2_50': False,
        'resnet_v2_101': False,
        'mobilenet_v1': False,
        'mobilenet_v1_075': False,
        'mobilenet_v1_050': True,
        'mobilenet_v1_025': True,
    }
    fb_name_lst = ['mobilenet_v1', 'mobilenet_v1_075', 'mobilenet_v1_050', 'mobilenet_v1_025',
                      'resnet_v1_50', 'resnet_v1_101', 'resnet_v2_50', 'resnet_v2_101',
                      'inception_v1', 'inception_v2']

    dir_path = FLAGS.ckpt_dir
    ckpt_files = [".".join(f.split(".")[:-1]) for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f)) and '.data' in f]
    for ckpt_file in ckpt_files:
        for fb_name in fb_name_lst:
            eval_ckpt_dir = 'checkpoint_eval/{}/{}/{}'.format(dir_path.split('/')[-1], ckpt_file.split('.')[-1], fb_name)
            if not os.path.exists(eval_ckpt_dir):
                os.makedirs(eval_ckpt_dir)
            run_training(fd_ckpt_file = os.path.join(dir_path, ckpt_file), ckpt_dir = eval_ckpt_dir, fb_name = fb_name, max_steps = model_max_steps_map[fb_name],
                     train_from_scratch = model_train_from_scratch_map[fb_name], ckpt_path = ckpt_path_map[fb_name])
            run_testing(ckpt_dir = eval_ckpt_dir, fb_name = fb_name, is_training=True)
            run_testing(ckpt_dir = eval_ckpt_dir, fb_name = fb_name, is_training=False)

if __name__ == '__main__':
    tf.app.run()
