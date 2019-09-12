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

from common_flags import COMMON_FLAGS
from nets import nets_factory
from data_preparation.VISPR.utils import *
from utils import *
from sklearn.metrics import average_precision_score
from fb_flags import FLAGS

slim = tf.contrib.slim
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ["CUDA_VISIBLE_DEVICES"]=FLAGS.GPU_id


def placeholder_inputs(batch_size):
    images_placeholder = tf.placeholder(tf.float32,
                                        shape=(batch_size, COMMON_FLAGS.CROP_HEIGHT, COMMON_FLAGS.CROP_WIDTH, COMMON_FLAGS.NCHANNEL))
    labels_placeholder = tf.placeholder(tf.float32, shape=(batch_size, COMMON_FLAGS.NUM_CLASSES_BUDGET))
    isTraining_placeholder = tf.placeholder(tf.bool)
    return images_placeholder, labels_placeholder, isTraining_placeholder

def create_architecture(scope, loss_fb_lst_dct, logits_fb_lst_dct, fb_dct, batch_size, images, labels, factor):
    loss_fb_images = 0.0
    logits_fb_images = tf.zeros([batch_size, COMMON_FLAGS.NUM_CLASSES_BUDGET])
    for name, fb in fb_dct.items():
        print(name)
        with tf.variable_scope(tf.get_variable_scope(), reuse=False):
            images = bilinear_resize(images, factor)
            logits, _ = fb(images)
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
        logits_fb_images += logits
        logits_fb_lst_dct[name].append(logits)
        loss_fb_lst_dct[name].append(loss)
        loss_fb_images += loss
    loss_fb_images_op = tf.divide(loss_fb_images, 4.0, 'LossFbMean')
    logits_fb_images_op = tf.divide(logits_fb_images, 4.0, 'LogitsFbMean')

    return loss_fb_images_op, logits_fb_images_op

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

def build_graph(gpu_num, batch_size, fb_name_lst, factor=1, num_epochs=None):
    from collections import defaultdict
    logits_fb_lst_dct = defaultdict(list)
    loss_fb_lst_dct = defaultdict(list)
    graph = tf.Graph()
    with graph.as_default():
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        images_placeholder, labels_placeholder, isTraining_placeholder = placeholder_inputs(gpu_num*batch_size)
        tower_grads = []
        logits_lst = []
        loss_lst = []
        # learning_rate = tf.train.exponential_decay(
        #     0.001,  # Base learning rate.
        #     global_step,  # Current index into the dataset.
        #     5000,  # Decay step.
        #     0.96,  # Decay rate.
        #     staircase=True)
        # # Use simple momentum for the optimization.
        # # opt = tf.train.MomentumOptimizer(learning_rate, 0.9)
        opt = tf.train.AdamOptimizer(1e-3)

        fb_dct = {}
        for name in fb_name_lst:
            fb_dct[name] = nets_factory.get_network_fn(
                name,
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
                        loss, logits = create_architecture(scope, loss_fb_lst_dct, logits_fb_lst_dct, fb_dct, batch_size, images, labels, factor)
                        logits_lst.append(logits)
                        loss_lst.append(loss)
                        varlist = tf.trainable_variables()
                        grads = opt.compute_gradients(loss, varlist)
                        tower_grads.append(grads)
                        # Reuse variables for the next tower.
                        tf.get_variable_scope().reuse_variables()

        loss_op = tf.reduce_mean(loss_lst, name='softmax')
        logits_op = tf.concat(logits_lst, 0)
        logits_op_lst = []
        for name in fb_name_lst:
            logits_op_lst.append(tf.concat(logits_fb_lst_dct[name], axis=0))

        zero_ops, accum_ops, apply_gradient_op = create_grad_accum_for_late_update(opt, tower_grads, varlist, FLAGS.n_minibatches, global_step, update_ops_depend=True, decay_with_global_step=False)

        tr_images_op, tr_labels_op = create_images_reading_ops(is_train=True, is_val=False, GPU_NUM=gpu_num, BATCH_SIZE=batch_size, NUM_EPOCHS=num_epochs)
        val_images_op, val_labels_op = create_images_reading_ops(is_train=False, is_val=True, GPU_NUM=gpu_num, BATCH_SIZE=batch_size, NUM_EPOCHS=num_epochs)
        test_images_op, test_labels_op = create_images_reading_ops(is_train=False, is_val=True, GPU_NUM=gpu_num, BATCH_SIZE=batch_size, NUM_EPOCHS=num_epochs)
        init_op = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())

        varlist_bn = [g for g in tf.global_variables() if 'moving_mean' in g.name]
        varlist_bn += [g for g in tf.global_variables() if 'moving_variance' in g.name]
        return (graph, init_op,
                zero_ops, accum_ops, apply_gradient_op,
                loss_op, logits_op, logits_op_lst,
                tr_images_op, tr_labels_op,
                val_images_op, val_labels_op,
                test_images_op, test_labels_op,
                images_placeholder, labels_placeholder, isTraining_placeholder,
                varlist, varlist_bn)

def train_fb(factor=1):
    # Create model directory
    ckpt_dir = os.path.join(FLAGS.ckpt_dir, str(factor))
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    fb_name_lst = ['resnet_v1_50', 'resnet_v2_50', 'mobilenet_v1', 'mobilenet_v1_075']

    (graph, init_op,
     zero_ops, accum_ops, apply_gradient_op,
     loss_op, logits_op, logits_op_lst,
     tr_images_op, tr_labels_op,
     val_images_op, val_labels_op,
     test_images_op, test_labels_op,
     images_placeholder, labels_placeholder, isTraining_placeholder,
     varlist, varlist_bn) = build_graph(FLAGS.GPU_NUM, FLAGS.batch_size, fb_name_lst, factor=factor, num_epochs=None)

    use_pretrained_model = True

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True

    with tf.Session(graph=graph, config=config) as sess:
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        if use_pretrained_model:
            fb_varlist_dict = {}
            model_name_mapping = {'resnet_v1_50': 'resnet_v1_50', 'resnet_v2_50': 'resnet_v2_50',
                                  'mobilenet_v1': 'MobilenetV1_1.0', 'mobilenet_v1_075': 'MobilenetV1_0.75'}
            for name in fb_name_lst:
                print(name)
                fb_varlist = [v for v in tf.trainable_variables() if
                           any(x in v.name for x in [model_name_mapping[name]])]
                fb_varlist_dict[name] = [v for v in fb_varlist if not any(x in v.name for x in ["logits"])]
                # print(varlist_dict[model_name])
                restore_from_model_zoo_ckpt(sess, os.path.join(COMMON_FLAGS.hdd_dir, 'model_zoo', COMMON_FLAGS.ckpt_path_map[name]), fb_varlist_dict[name], name)
        else:
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir=ckpt_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver = tf.train.Saver(tf.trainable_variables())
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Session restored from model at {}!'.format(
                    ckpt.model_checkpoint_path))
            else:
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), ckpt_dir)

        saver = tf.train.Saver(varlist + varlist_bn)
        for step in range(FLAGS.max_steps):

            loss_summary = update_fb(sess, step, FLAGS.n_minibatches, zero_ops, apply_gradient_op, accum_ops, loss_op,
                      tr_images_op, tr_labels_op, images_placeholder, labels_placeholder,
                      isTraining_placeholder)
            print(loss_summary)

            if step % FLAGS.val_step == 0:
                eval_summary = eval_fb(sess, step, FLAGS.n_minibatches, logits_op, loss_op, tr_images_op, tr_labels_op,
                            images_placeholder, labels_placeholder, isTraining_placeholder)
                print("TRAINING: "+eval_summary)

                eval_summary = eval_fb(sess, step, FLAGS.n_minibatches, logits_op, loss_op, val_images_op, val_labels_op,
                            images_placeholder, labels_placeholder, isTraining_placeholder)
                print("VALIDATION: "+eval_summary)

            # Save a checkpoint and evaluate the model periodically.
            if step % FLAGS.save_step == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_path = os.path.join(ckpt_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

        coord.request_stop()
        coord.join(threads)

def test_fb(factor=1, is_training=False):
    ckpt_dir = os.path.join(FLAGS.ckpt_dir, str(factor))

    fb_name_lst = ['resnet_v1_50', 'resnet_v2_50', 'mobilenet_v1', 'mobilenet_v1_075']

    (graph, init_op,
     zero_ops, accum_ops, apply_gradient_op,
     loss_op, logits_op, logits_op_lst,
     tr_images_op, tr_labels_op,
     val_images_op, val_labels_op,
     test_images_op, test_labels_op,
     images_placeholder, labels_placeholder, isTraining_placeholder,
     varlist, varlist_bn) = build_graph(FLAGS.GPU_NUM, FLAGS.batch_size, fb_name_lst, factor=factor, num_epochs=1)

    if is_training:
        images_op, labels_op = tr_images_op, tr_labels_op
    else:
        images_op, labels_op = test_images_op, test_labels_op,

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True
    with tf.Session(graph=graph, config=config) as sess:
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        saver = tf.train.Saver(varlist + varlist_bn)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir=ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Session restored from pretrained model at {}!'.format(ckpt.model_checkpoint_path))
        else:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), ckpt_dir)

        fb_name_lst += ['ensemble']
        loss_fb_lst = []
        pred_probs_lst_lst = [[] for _ in range(len(fb_name_lst))]
        gt_lst = []
        try:
            while not coord.should_stop():
                images, labels = sess.run([images_op, labels_op])
                # write_video(videos, labels)
                gt_lst.append(labels)
                feed = {images_placeholder: images, labels_placeholder: labels,
                        isTraining_placeholder: True}

                value_lst = sess.run([loss_op, logits_op] + logits_op_lst, feed_dict=feed)
                loss_fb_lst.append(value_lst[0])
                for i in range(len(fb_name_lst)):
                    pred_probs_lst_lst[i].append(value_lst[i + 1])
                # print(tf.argmax(softmax_logits, 1).eval(session=sess))
        except tf.errors.OutOfRangeError:
            print('Done testing on all the examples')
        finally:
            coord.request_stop()
        coord.join(threads)
        sess.close()

        gt_mat = np.concatenate(gt_lst, axis=0)
        n_examples, n_labels = gt_mat.shape
        for i in range(len(fb_name_lst)):
            save_dir = os.path.join(ckpt_dir, 'evaluation')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            isTraining = lambda bool: "training" if bool else "testing"
            with open(os.path.join(save_dir, '{}_class_scores_{}.txt'.format(fb_name_lst[i],
                                                                             isTraining(is_training))), 'w') as wf:
                pred_probs_mat = np.concatenate(pred_probs_lst_lst[i], axis=0)
                wf.write('# Examples = {}\n'.format(n_examples))
                wf.write('# Labels = {}\n'.format(n_labels))
                wf.write('Average Loss = {}\n'.format(np.mean(loss_fb_lst)))
                wf.write("Macro MAP = {:.2f}\n".format(
                    100 * average_precision_score(gt_mat, pred_probs_mat, average='macro')))
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
    train_fb(factor=FLAGS.factor)
    test_fb(factor=FLAGS.factor, is_training=False)
    test_fb(factor=FLAGS.factor, is_training=True)
    # for factor in [2, 4, 6, 8, 14, 16]:

if __name__ == '__main__':
  tf.app.run()
