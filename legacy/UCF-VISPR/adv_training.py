#!/usr/bin/env python

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ["CUDA_VISIBLE_DEVICES"]="4,5,6,7"
import sys
sys.path.insert(0, '..')

import time
from six.moves import xrange
import tensorflow as tf
import numpy as np
from tf_flags import FLAGS
from nets import nets_factory
#import cv2
import errno
import pprint
import itertools
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage import img_as_float
from skimage import exposure
matplotlib.rcParams['font.size'] = 8
from sklearn.metrics import average_precision_score
from vispr.utils import load_attributes, labels_to_vec
from degradNet import residualNet
from utilityNet import C3DNet
from input_data import *

def placeholder_inputs(batch_size):
    videos_placeholder = tf.placeholder(tf.float32, shape=(batch_size, FLAGS.depth, None, None, FLAGS.nchannel))
    action_labels_placeholder = tf.placeholder(tf.int64, shape=(batch_size))
    actor_labels_placeholder = tf.placeholder(tf.int64, shape=(batch_size))
    dropout_placeholder = tf.placeholder(tf.float32)
    isTraining_placeholder = tf.placeholder(tf.bool)
    return videos_placeholder, action_labels_placeholder, actor_labels_placeholder, dropout_placeholder, isTraining_placeholder

def tower_loss_mse(preds, labels):
    labels = tf.cast(labels, tf.float32)
    MSE = tf.reduce_mean(
            tf.square(labels - preds))
    return MSE

def tower_loss_xentropy_dense(logits, labels):
    labels = tf.cast(labels, tf.float32)
    cross_entropy_mean = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    )
    return cross_entropy_mean

def tower_loss_xentropy_sparse(name_scope, logits, labels, use_weight_decay=False):
    labels = tf.cast(labels, tf.int64)
    cross_entropy_mean = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))

    if use_weight_decay:
        tf.add_to_collection('c3d_losses', cross_entropy_mean)
        losses = tf.get_collection('c3d_losses', name_scope)
        return tf.add_n(losses, name='c3d_losses')
    return cross_entropy_mean

def tower_loss_neg_entropy(logits):
    softmax = tf.nn.softmax(logits)
    nentropy = tf.reduce_sum(tf.multiply(softmax, tf.log(softmax)))
    return nentropy

def tower_loss_max_neg_entropy(logits_lst):
    nentropy_tensor_lst = []
    for logits in logits_lst:
        softmax = tf.nn.softmax(logits)
        nentropy_tensor = tf.reduce_sum(tf.multiply(softmax, tf.log(softmax)), 1)
        nentropy_tensor_lst.append(nentropy_tensor)
    nentropy_tensor_stack = tf.stack(nentropy_tensor_lst, axis=0)
    argmax_nentropy = tf.argmax(nentropy_tensor_stack, axis=0)
    max_nentropy = tf.reduce_mean(tf.reduce_max(nentropy_tensor_stack, axis=0))
    return max_nentropy, argmax_nentropy

def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
        List of pairs of (gradient, variable) where the gradient has been averaged
        across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

def accuracy(logits, labels):
    correct_pred = tf.equal(tf.argmax(tf.nn.softmax(logits), axis=1), labels)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return accuracy

def run_pretraining_residual():
    # Create model directory
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    use_pretrained_model = True

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True
    with tf.Graph().as_default():
        with tf.Session(config=config) as sess:
            global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
            videos_placeholder = tf.placeholder(tf.float32, shape=(FLAGS.video_batch_size * FLAGS.gpu_num, FLAGS.depth, None, None, FLAGS.nchannel))
            images_placeholder = tf.placeholder(tf.float32, shape=(FLAGS.image_batch_size * FLAGS.gpu_num, None, None, FLAGS.nchannel))
            utility_labels_placeholder = tf.placeholder(tf.int64, shape=(FLAGS.video_batch_size * FLAGS.gpu_num))
            budget_labels_placeholder = tf.placeholder(tf.float32, shape=(FLAGS.image_batch_size * FLAGS.gpu_num, FLAGS.num_classes_budget))
            dropout_placeholder = tf.placeholder(tf.float32)
            isTraining_placeholder = tf.placeholder(tf.bool)

            tower_grads_degradation = []
            tower_grads_utility_main = []
            tower_grads_utility_finetune = []
            tower_grads_budget = []

            logits_utility_lst = []
            logits_budget_images_lst = []

            loss_utility_lst = []
            loss_budget_images_lst = []

            opt_degradation = tf.train.AdamOptimizer(1e-3)
            opt_utility_finetune = tf.train.AdamOptimizer(1e-4)
            opt_utility = tf.train.AdamOptimizer(1e-5)
            learning_rate = tf.train.exponential_decay(
                0.001,  # Base learning rate.
                global_step,  # Current index into the dataset.
                5000,  # Decay step.
                0.96,  # Decay rate.
                staircase=True)
            # Use simple momentum for the optimization.
            opt_budget = tf.train.MomentumOptimizer(learning_rate, 0.9)
            budgetNet = nets_factory.get_network_fn(
                FLAGS.model_name,
                num_classes=FLAGS.num_classes_budget,
                weight_decay=FLAGS.weight_decay,
                is_training=isTraining_placeholder)
            with tf.variable_scope(tf.get_variable_scope()) as scope:
                for gpu_index in range(0, FLAGS.gpu_num):
                    with tf.device('/gpu:%d' % gpu_index):
                        print('/gpu:%d' % gpu_index)
                        with tf.name_scope('%s_%d' % ('gpu', gpu_index)) as scope:
                            videos = tf.reshape(videos_placeholder[gpu_index * FLAGS.video_batch_size:(gpu_index + 1) * FLAGS.video_batch_size],
                                                [FLAGS.video_batch_size * FLAGS.depth, FLAGS.crop_height, FLAGS.crop_width, FLAGS.nchannel])
                            videos = residualNet(videos)
                            videos_utility = tf.reshape(videos,
                                                [FLAGS.video_batch_size, FLAGS.depth, FLAGS.crop_height, FLAGS.crop_width, FLAGS.nchannel])
                            logits_utility = C3DNet(videos_utility, dropout_placeholder)
                            logits_utility_lst.append(logits_utility)
                            loss_utility = tower_loss_xentropy_sparse(
                                scope,
                                logits_utility,
                                utility_labels_placeholder[gpu_index * FLAGS.video_batch_size:(gpu_index + 1) * FLAGS.video_batch_size],
                                use_weight_decay = True
                                )
                            loss_utility_lst.append(loss_utility)
                            with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                                LR_images = residualNet(images_placeholder[gpu_index * FLAGS.image_batch_size:(gpu_index + 1) * FLAGS.image_batch_size])
                            logits_budget_images, _ = budgetNet(LR_images)
                            logits_budget_images_lst.append(logits_budget_images)
                            loss_budget_images = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = logits_budget_images,
                                                labels = budget_labels_placeholder[gpu_index * FLAGS.image_batch_size:(gpu_index + 1) * FLAGS.image_batch_size, :]))
                            loss_budget_images_lst.append(loss_budget_images)

                            varlist_degradtion = [v for v in tf.trainable_variables() if any(x in v.name for x in ["DegradationModule"])]
                            print([v.name for v in varlist_degradtion])
                            varlist_utility = [v for v in tf.trainable_variables() if any(x in v.name for x in ["UtilityModule"])]
                            print([v.name for v in varlist_utility])
                            varlist_utility_finetune = [v for v in varlist_utility if any(x in v.name.split('/')[1] for x in ["out", "d2"])]
                            print([v.name for v in varlist_utility_finetune])
                            varlist_budget = [v for v in tf.trainable_variables() if not any(x in v.name for x in ["UtilityModule", "DegradationModule"])]
                            print([v.name for v in varlist_budget])
                            varlist_utility_main = list(set(varlist_utility) - set(varlist_utility_finetune))
                            print([v.name for v in varlist_utility_main])

                            grads_degradation = opt_degradation.compute_gradients(loss_utility, varlist_degradtion)
                            grads_utility_main = opt_utility.compute_gradients(loss_utility, varlist_utility_main)
                            grads_utility_finetune = opt_utility_finetune.compute_gradients(loss_utility, varlist_utility_finetune)
                            grads_budget = opt_budget.compute_gradients(loss_budget_images, varlist_budget)

                            tower_grads_degradation.append(grads_degradation)
                            tower_grads_budget.append(grads_budget)
                            tower_grads_utility_main.append(grads_utility_main)
                            tower_grads_utility_finetune.append(grads_utility_finetune)

                            tf.get_variable_scope().reuse_variables()

            loss_utility_op = tf.reduce_mean(loss_utility_lst, name='softmax')
            loss_budget_op = tf.reduce_mean(loss_budget_images_lst, name='softmax')

            logits_utility_op = tf.concat(logits_utility_lst, 0)
            logits_budget_images_op = tf.concat(logits_budget_images_lst, 0)
            accuracy_util = accuracy(logits_utility_op, utility_labels_placeholder)

            grads_degradation = average_gradients(tower_grads_degradation)
            grads_utility_main = average_gradients(tower_grads_utility_main)
            grads_utility_finetune = average_gradients(tower_grads_utility_finetune)
            grads_budget = average_gradients(tower_grads_budget)

            with tf.device('/cpu:%d' % 0):
                tvs_degradation = varlist_degradtion
                accum_vars_degradtion =  [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in tvs_degradation]
                zero_ops_degradation = [tv.assign(tf.zeros_like(tv)) for tv in accum_vars_degradtion]

            with tf.device('/cpu:%d' % 0):
                tvs_budget = varlist_budget
                accum_vars_budget =  [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in tvs_budget]
                zero_ops_budget = [tv.assign(tf.zeros_like(tv)) for tv in accum_vars_budget]

            with tf.device('/cpu:%d' % 0):
                tvs_utility_finetune = varlist_utility_finetune
                accum_vars_utility_finetune =  [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in tvs_utility_finetune]
                zero_ops_utility_finetune = [tv.assign(tf.zeros_like(tv)) for tv in accum_vars_utility_finetune]

            with tf.device('/cpu:%d' % 0):
                tvs_utility_main = varlist_utility_main
                accum_vars_utility_main =  [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in tvs_utility_main]
                zero_ops_utility_main = [tv.assign(tf.zeros_like(tv)) for tv in accum_vars_utility_main]


            accum_ops_degradation = [accum_vars_degradtion[i].assign_add(gv[0]/FLAGS.n_minibatches) for i, gv in enumerate(grads_degradation)]
            accum_ops_utility_main = [accum_vars_utility_main[i].assign_add(gv[0]/FLAGS.n_minibatches) for i, gv in enumerate(grads_utility_main)]
            accum_ops_utility_finetune = [accum_vars_utility_finetune[i].assign_add(gv[0]/FLAGS.n_minibatches) for i, gv in enumerate(grads_utility_finetune)]
            accum_ops_budget = [accum_vars_budget[i].assign_add(gv[0]/FLAGS.n_minibatches) for i, gv in enumerate(grads_budget)]


            apply_gradient_op_degradation = opt_degradation.apply_gradients([(accum_vars_degradtion[i].value(), gv[1]) for i, gv in enumerate(grads_degradation)], global_step=global_step)
            apply_gradient_op_utility_main = opt_utility.apply_gradients([(accum_vars_utility_main[i].value(), gv[1]) for i, gv in enumerate(grads_utility_main)], global_step=global_step)
            apply_gradient_op_utility_finetune = opt_utility.apply_gradients([(accum_vars_utility_finetune[i].value(), gv[1]) for i, gv in enumerate(grads_utility_finetune)], global_step=global_step)
            apply_gradient_op_budget = opt_budget.apply_gradients([(accum_vars_budget[i].value(), gv[1]) for i, gv in enumerate(grads_budget)], global_step=global_step)


            train_video_files = [os.path.join(FLAGS.train_videos_files_dir, f) for f in
                           os.listdir(FLAGS.train_videos_files_dir) if f.endswith('.tfrecords')]
            val_video_files = [os.path.join(FLAGS.val_videos_files_dir, f) for f in
                         os.listdir(FLAGS.val_videos_files_dir) if f.endswith('.tfrecords')]
            print(train_video_files)
            print(val_video_files)

            train_image_files = [os.path.join(FLAGS.train_images_files_dir, f) for f in
                           os.listdir(FLAGS.train_images_files_dir) if f.endswith('.tfrecords')]
            print(train_image_files)
            val_image_files = [os.path.join(FLAGS.val_images_files_dir, f) for f in
                           os.listdir(FLAGS.val_images_files_dir) if f.endswith('.tfrecords')]
            print(val_image_files)
            test_image_files = [os.path.join(FLAGS.test_images_files_dir, f) for f in
                           os.listdir(FLAGS.test_images_files_dir) if f.endswith('.tfrecords')]
            print(test_image_files)

            tr_videos_op, tr_videos_labels_op = inputs_videos(filenames = train_video_files,
                                                 batch_size=FLAGS.video_batch_size * FLAGS.gpu_num,
                                                 num_epochs=None,
                                                 num_threads=FLAGS.num_threads,
                                                 num_examples_per_epoch=FLAGS.num_examples_per_epoch,
                                                 shuffle=True)
            val_videos_op, val_videos_labels_op = inputs_videos(filenames = val_video_files,
                                                   batch_size=FLAGS.video_batch_size * FLAGS.gpu_num,
                                                   num_epochs=None,
                                                   num_threads=FLAGS.num_threads,
                                                   num_examples_per_epoch=FLAGS.num_examples_per_epoch,
                                                   shuffle=True)
            tr_images_op, tr_images_labels_op = inputs_images(filenames = train_image_files + val_image_files,
                                                   batch_size=FLAGS.image_batch_size * FLAGS.gpu_num,
                                                   num_epochs=None,
                                                   num_threads=FLAGS.num_threads,
                                                   num_examples_per_epoch=FLAGS.num_examples_per_epoch,
                                                   shuffle=True)
            val_images_op, val_images_labels_op = inputs_images(filenames = test_image_files,
                                                   batch_size=FLAGS.image_batch_size * FLAGS.gpu_num,
                                                   num_epochs=None,
                                                   num_threads=FLAGS.num_threads,
                                                   num_examples_per_epoch=FLAGS.num_examples_per_epoch,
                                                   shuffle=True)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            init_op = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())
            sess.run(init_op)

            def restore_model_ckpt(ckpt_dir, varlist, modulename):
                ckpt = tf.train.get_checkpoint_state(checkpoint_dir=ckpt_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    print("##########################{}##########################".format(modulename))
                    print(varlist)
                    saver = tf.train.Saver(varlist)
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    print(
                        'Session restored from pretrained model at {}!'.format(ckpt.model_checkpoint_path))
                else:
                    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), ckpt_dir)

            def restore_model_pretrained_C3D():
                if os.path.isfile(FLAGS.pretrained_C3D):
                    varlist = [v for v in tf.trainable_variables() if
                                       any(x in v.name for x in ["UtilityModule"])]
                    varlist = [v for v in varlist if not any(x in v.name.split('/')[1] for x in ["out", "d2"])]
                    vardict = {v.name[:-2].replace('UtilityModule', 'var_name'):v for v in varlist}
                    for key, value in vardict.items():
                        print(key)
                    saver = tf.train.Saver(vardict)
                    saver.restore(sess, FLAGS.pretrained_C3D)
                    print('#############################Session restored from pretrained model at {}!#############################'.format(FLAGS.pretrained_C3D))
                else:
                    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), FLAGS.pretrained_C3D)

            # Create a saver for writing training checkpoints.
            if use_pretrained_model:
                varlist = [v for v in tf.trainable_variables() if
                           any(x in v.name.split('/')[0] for x in ["DegradationModule"])]
                restore_model_ckpt(FLAGS.degradation_models, varlist, "DegradationModule")
                restore_model_pretrained_C3D()
                varlist = [v for v in tf.trainable_variables() if
                                  not any(x in v.name for x in ["UtilityModule", "DegradationModule"])]
                #restore_model_ckpt(FLAGS.budget_models, varlist, "BudgetModule")
            else:
                saver = tf.train.Saver()
                ckpt = tf.train.get_checkpoint_state(checkpoint_dir=FLAGS.checkpoint_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    print('Session restored from trained model at {}!'.format(ckpt.model_checkpoint_path))
                else:
                    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), FLAGS.checkpoint_dir)


            gvar_list = tf.global_variables()
            bn_moving_vars = [g for g in gvar_list if 'moving_mean' in g.name]
            bn_moving_vars += [g for g in gvar_list if 'moving_variance' in g.name]
            #saver = tf.train.Saver(tf.trainable_variables() + bn_moving_vars)
            saver = tf.train.Saver()
            for step in xrange(FLAGS.pretraining_steps_utility):
                start_time = time.time()
                sess.run([zero_ops_utility_finetune, zero_ops_utility_main, zero_ops_degradation])
                loss_utility_lst = []
                for _ in itertools.repeat(None, FLAGS.n_minibatches):
                    tr_videos, tr_videos_labels = sess.run(
                            [tr_videos_op, tr_videos_labels_op])
                    _, _, _, loss_utility = sess.run([accum_ops_utility_finetune, accum_ops_utility_main,
                                                                       accum_ops_degradation, loss_utility_op],
                            feed_dict={videos_placeholder: tr_videos,
                                       utility_labels_placeholder: tr_videos_labels,
                                       dropout_placeholder: 1.0,
                                       })
                    loss_utility_lst.append(loss_utility)
                    sess.run([apply_gradient_op_utility_finetune, apply_gradient_op_utility_main])
                    loss_summary = 'Utility Module + Degradation Module, Step: {:4d}, time: {:.4f}, utility loss: {:.8f}'.format(
                        step,
                        time.time() - start_time,
                        np.mean(loss_utility_lst))
                    print(loss_summary)

                if step % FLAGS.val_step == 0:
                    start_time = time.time()
                    acc_util_train_lst, loss_utility_train_lst = [], []
                    for _ in itertools.repeat(None, 30):
                        tr_videos, tr_videos_labels = sess.run(
                            [tr_videos_op, tr_videos_labels_op])
                        acc_util, loss_utility = sess.run(
                            [accuracy_util, loss_utility_op],
                            feed_dict={videos_placeholder: tr_videos,
                                       utility_labels_placeholder: tr_videos_labels,
                                       dropout_placeholder: 1.0,
                                       })
                        acc_util_train_lst.append(acc_util)
                        loss_utility_train_lst.append(loss_utility)

                    train_summary = "Step: {:4d}, time: {:.4f}, utility loss: {:.8f}, training utility accuracy: {:.5f}".format(
                        step,
                        time.time() - start_time,
                        np.mean(loss_utility_train_lst), np.mean(acc_util_train_lst))
                    print(train_summary)

                    start_time = time.time()
                    acc_util_val_lst,  loss_utility_val_lst = [], []
                    for _ in itertools.repeat(None, 30):
                        val_videos, val_videos_labels = sess.run(
                            [val_videos_op, val_videos_labels_op])
                        acc_util, loss_utility = sess.run(
                            [accuracy_util, loss_utility_op],
                            feed_dict={videos_placeholder: val_videos,
                                       utility_labels_placeholder: val_videos_labels,
                                       dropout_placeholder: 1.0,
                                       })
                        acc_util_val_lst.append(acc_util)
                        loss_utility_val_lst.append(loss_utility)

                    test_summary = "Step: {:4d}, time: {:.4f}, utility loss: {:.8f}, validation utility accuracy: {:.5f}".format(
                        step,
                        time.time() - start_time,
                        np.mean(loss_utility_val_lst), np.mean(acc_util_val_lst))
                    print(test_summary)

                    loss_budget_lst = []
                    pred_probs_lst = []
                    gt_lst = []
                    for _ in itertools.repeat(None, 30):
                        val_images, val_images_labels = sess.run(
                                [val_images_op, val_images_labels_op])
                        gt_lst.append(val_images_labels)
                        logits_budget, loss_budget = sess.run([logits_budget_images_op, loss_budget_op],
                                feed_dict={images_placeholder: val_images,
                                           budget_labels_placeholder: val_images_labels,
                                           dropout_placeholder: 1.0,
                                           isTraining_placeholder: True})
                        loss_budget_lst.append(loss_budget)
                        pred_probs_lst.append(logits_budget)

                    pred_probs_mat = np.concatenate(pred_probs_lst, axis=0)
                    gt_mat = np.concatenate(gt_lst, axis=0)
                    n_examples, n_labels = gt_mat.shape
                    print('# Examples = ', n_examples)
                    print('# Labels = ', n_labels)
                    print('Macro MAP = {:.2f}'.format(100 * average_precision_score(gt_mat, pred_probs_mat, average='macro')))

                if step % FLAGS.save_step == 0 or (step + 1) == FLAGS.max_steps:
                    checkpoint_path = os.path.join(FLAGS.checkpoint_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)

            for step in xrange(FLAGS.pretraining_steps_budget):
                start_time = time.time()
                sess.run(zero_ops_budget)
                loss_budget_lst = []
                for _ in itertools.repeat(None, FLAGS.n_minibatches):
                    tr_images, tr_images_labels = sess.run([tr_images_op, tr_images_labels_op])

                    _, loss_budget = sess.run([accum_ops_budget, loss_budget_op],
                                        feed_dict={images_placeholder: tr_images,
                                                   budget_labels_placeholder: tr_images_labels,
                                                   dropout_placeholder: 1.0,
                                                   isTraining_placeholder: True})
                    loss_budget_lst.append(loss_budget)
                sess.run([apply_gradient_op_budget])
                loss_summary = 'Budget Module, Step: {:4d}, time: {:.4f}, budget loss: {:.8f}'.format(step,
                                                                                                      time.time() - start_time,
                                                                                                      np.mean(loss_budget_lst))
                print(loss_summary)

                if step % FLAGS.val_step == 0:
                    loss_budget_lst = []
                    pred_probs_lst = []
                    gt_lst = []
                    for _ in itertools.repeat(None, 30):
                        val_images, val_images_labels = sess.run([val_images_op, val_images_labels_op])
                        gt_lst.append(val_images_labels)
                        logits_budget, loss_budget = sess.run([logits_budget_images_op, loss_budget_op],
                                                            feed_dict={images_placeholder: val_images,
                                                                       budget_labels_placeholder: val_images_labels,
                                                                       dropout_placeholder: 1.0,
                                                                       isTraining_placeholder: True})
                        loss_budget_lst.append(loss_budget)
                        pred_probs_lst.append(logits_budget)

                    pred_probs_mat = np.concatenate(pred_probs_lst, axis=0)
                    gt_mat = np.concatenate(gt_lst, axis=0)
                    n_examples, n_labels = gt_mat.shape
                    print('# Examples = ', n_examples)
                    print('# Labels = ', n_labels)
                    print('Macro MAP = {:.2f}'.format(100 * average_precision_score(gt_mat, pred_probs_mat, average='macro')))

                if step % FLAGS.save_step == 0 or (step + 1) == FLAGS.max_steps:
                    checkpoint_path = os.path.join(FLAGS.whole_pretrained_checkpoint_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)

            coord.request_stop()
            coord.join(threads)

        print("done")

def run_training_multi_model_residual():
    # Create model directory
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)

    use_pretrained_model = False
    use_whole_pretrained_model = False

    from collections import defaultdict
    logits_budget_images_lst_dct = defaultdict(list)
    loss_budget_images_lst_dct = defaultdict(list)

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True
    tf.reset_default_graph()
    with tf.Graph().as_default():
        with tf.Session(config=config) as sess:
            global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
            videos_placeholder = tf.placeholder(tf.float32, shape=(FLAGS.video_batch_size * FLAGS.gpu_num, FLAGS.depth, None, None, FLAGS.nchannel))
            images_placeholder = tf.placeholder(tf.float32, shape=(FLAGS.image_batch_size * FLAGS.gpu_num, None, None, FLAGS.nchannel))
            utility_labels_placeholder = tf.placeholder(tf.int64, shape=(FLAGS.video_batch_size * FLAGS.gpu_num))
            budget_images_labels_placeholder = tf.placeholder(tf.float32, shape=(FLAGS.image_batch_size * FLAGS.gpu_num, FLAGS.num_classes_budget))
            dropout_placeholder = tf.placeholder(tf.float32)
            budget_videos_labels_placeholder = tf.placeholder(tf.float32, shape=(FLAGS.video_batch_size * FLAGS.gpu_num, FLAGS.num_classes_budget))
            isTraining_placeholder = tf.placeholder(tf.bool)

            tower_grads_degradation = []
            tower_grads_utility = []
            tower_grads_budget = []

            logits_utility_lst = []
            logits_budget_images_lst = []

            loss_utility_lst = []
            loss_budget_images_lst = []
            loss_filter_lst = []
            argmax_centpy_lst = []
            opt_degradation = tf.train.AdamOptimizer(1e-4)
            opt_utility = tf.train.AdamOptimizer(1e-5)
            opt_budget = tf.train.AdamOptimizer(1e-4)

            budgetNet_model_name_lst = ['resnet_v1_50', 'resnet_v2_50', 'mobilenet_v1', 'mobilenet_v1_075']
            #budgetNet_model_name_lst = ['resnet_v1_50', 'resnet_v2_50', 'mobilenet_v1']
            #budgetNet_model_name_lst = ['resnet_v1_50', 'resnet_v2_50']
            #budgetNet_model_name_lst = ['resnet_v1_50']
            budgetNet_dict = {}
            for model_name in budgetNet_model_name_lst:
                budgetNet_dict[model_name] = nets_factory.get_network_fn(
                                    model_name,
                                    num_classes=FLAGS.num_classes_budget,
                                    weight_decay=FLAGS.weight_decay,
                                    is_training=isTraining_placeholder)
            with tf.variable_scope(tf.get_variable_scope()) as scope:
                for gpu_index in range(0, FLAGS.gpu_num):
                    with tf.device('/gpu:%d' % gpu_index):
                        print('/gpu:%d' % gpu_index)
                        with tf.name_scope('%s_%d' % ('gpu', gpu_index)) as scope:

                            videos = tf.reshape(videos_placeholder[gpu_index * FLAGS.video_batch_size:(gpu_index + 1) * FLAGS.video_batch_size],
                                                [FLAGS.video_batch_size * FLAGS.depth, FLAGS.crop_height, FLAGS.crop_width, FLAGS.nchannel])
                            with tf.variable_scope(tf.get_variable_scope(), reuse=False):
                                videos = residualNet(videos, training=False)
                            videos_utility = tf.reshape(videos,
                                                [FLAGS.video_batch_size, FLAGS.depth, FLAGS.crop_height, FLAGS.crop_width, FLAGS.nchannel])
                            logits_utility = C3DNet(videos_utility, dropout_placeholder)
                            logits_utility_lst.append(logits_utility)
                            loss_utility = tower_loss_xentropy_sparse(
                                scope,
                                logits_utility,
                                utility_labels_placeholder[gpu_index * FLAGS.video_batch_size:(gpu_index + 1) * FLAGS.video_batch_size],
                                use_weight_decay = True
                            )
                            loss_utility_lst.append(loss_utility)

                            with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                                LR_images = residualNet(images_placeholder[gpu_index * FLAGS.image_batch_size:(gpu_index + 1) * FLAGS.image_batch_size], training=False)

                            loss_budget_images = 0.0
                            logits_budget_images = tf.zeros([FLAGS.image_batch_size, FLAGS.num_classes_budget])
                            for model_name in budgetNet_model_name_lst:
                                print(model_name)
                                print(tf.trainable_variables())
                                with tf.variable_scope(tf.get_variable_scope(), reuse=False):
                                    logits, _ = budgetNet_dict[model_name](LR_images)
                                loss = tf.reduce_mean(
                                    tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                                                labels=budget_images_labels_placeholder[gpu_index * FLAGS.image_batch_size:(gpu_index + 1) * FLAGS.image_batch_size,:]))
                                logits_budget_images += logits
                                logits_budget_images_lst_dct[model_name].append(logits)
                                loss_budget_images_lst_dct[model_name].append(loss)
                                loss_budget_images += loss
                            loss_budget_images_lst.append(loss_budget_images)
                            logits_budget_images = tf.divide(logits_budget_images, 4.0, 'LogitsBudgetMean')
                            logits_budget_images_lst.append(logits_budget_images)
                            videos_budget = videos
                            loss_budget_videos = 0.0
                            loss_budget_videos_lst = []
                            for model_name in budgetNet_model_name_lst:
                                print(model_name)
                                with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                                    logits_budget_videos,_ = budgetNet_dict[model_name](videos_budget)
                                logits_budget_videos = tf.reshape(logits_budget_videos, [-1, FLAGS.depth, FLAGS.num_classes_budget])
                                logits_budget_videos = tf.reduce_mean(logits_budget_videos, axis=1, keep_dims=False)
                                loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_budget_videos,
                                                                labels=budget_videos_labels_placeholder[gpu_index * FLAGS.video_batch_size:(gpu_index + 1) * FLAGS.video_batch_size, :]))
                                loss_tensor = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_budget_videos,
                                                                labels=budget_videos_labels_placeholder[gpu_index * FLAGS.video_batch_size:(gpu_index + 1) * FLAGS.video_batch_size, :])
                                print(loss_tensor.shape)
                                loss_tensor = tf.reshape(tf.reduce_mean(loss_tensor, axis=1), [-1])
                                loss_budget_videos += loss
                                print(loss_tensor.shape)
                                loss_budget_videos_lst.append(loss_tensor)

                            loss_budget_videos_tensor_stack = tf.stack(loss_budget_videos_lst, axis=0)
                            print('##############################################################')
                            print(loss_budget_videos_tensor_stack.shape)
                            print(tf.reduce_max(loss_budget_videos_tensor_stack, axis=0).shape)
                            print('##############################################################')
                            argmax_centpy = tf.argmax(loss_budget_videos_tensor_stack, axis=0)
                            argmax_centpy_lst.append(argmax_centpy)

                            min_centpy = tf.reduce_mean(tf.reduce_max(loss_budget_videos_tensor_stack, axis=0))
                            loss_filter = loss_utility + FLAGS.lambda_ * min_centpy
                            loss_filter_lst.append(loss_filter)

                            varlist_degrad = [v for v in tf.trainable_variables() if not any(x in v.name for x in ["UtilityModule", "MobilenetV1_1.0", "MobilenetV1_0.75", "resnet_v1_50", "resnet_v2_50"])]
                            print([v.name for v in varlist_degrad])
                            varlist_utility = [v for v in tf.trainable_variables() if any(x in v.name for x in ["UtilityModule"])]
                            print([v.name for v in varlist_utility])
                            varlist_budget = [v for v in tf.trainable_variables() if any(x in v.name for x in ["MobilenetV1_1.0", "MobilenetV1_0.75", "resnet_v1_50", "resnet_v2_50"])]
                            print([v.name for v in varlist_budget])

                            grads_degradation = opt_degradation.compute_gradients(loss_filter, varlist_degrad)
                            grads_utility = opt_utility.compute_gradients(loss_utility, varlist_utility+varlist_degrad)
                            grads_budget = opt_budget.compute_gradients(loss_budget_images, varlist_budget)

                            tower_grads_degradation.append(grads_degradation)
                            tower_grads_budget.append(grads_budget)
                            tower_grads_utility.append(grads_utility)

                            tf.get_variable_scope().reuse_variables()

            argmax_cent_op = tf.concat(argmax_centpy_lst, 0)

            loss_utility_op = tf.reduce_mean(loss_utility_lst, name='softmax')
            loss_budget_op = tf.reduce_mean(loss_budget_images_lst, name='softmax')
            loss_filter_op = tf.reduce_mean(loss_filter_lst, name='softmax')

            logits_utility_op = tf.concat(logits_utility_lst, 0)
            logits_budget_images_op = tf.concat(logits_budget_images_lst, 0)
            accuracy_util = accuracy(logits_utility_op, utility_labels_placeholder)

            grads_degradation = average_gradients(tower_grads_degradation)
            grads_utility = average_gradients(tower_grads_utility)
            grads_budget = average_gradients(tower_grads_budget)

            with tf.device('/cpu:%d' % 0):
                tvs_degradation = varlist_degrad
                accum_vars_degradtion =  [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in tvs_degradation]
                zero_ops_degradation = [tv.assign(tf.zeros_like(tv)) for tv in accum_vars_degradtion]

            with tf.device('/cpu:%d' % 0):
                tvs_budget = varlist_budget
                accum_vars_budget =  [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in tvs_budget]
                zero_ops_budget = [tv.assign(tf.zeros_like(tv)) for tv in accum_vars_budget]

            with tf.device('/cpu:%d' % 0):
                tvs_utility = varlist_utility+varlist_degrad
                print(tvs_utility)
                print('###########################################################')
                print(grads_utility)
                accum_vars_utility =  [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in tvs_utility]
                zero_ops_utility = [tv.assign(tf.zeros_like(tv)) for tv in accum_vars_utility]
            accum_ops_degradation = [accum_vars_degradtion[i].assign_add(gv[0]/FLAGS.n_minibatches) for i, gv in enumerate(grads_degradation)]
            accum_ops_utility = [accum_vars_utility[i].assign_add(gv[0]/FLAGS.n_minibatches) for i, gv in enumerate(grads_utility)]
            accum_ops_budget = [accum_vars_budget[i].assign_add(gv[0]/FLAGS.n_minibatches) for i, gv in enumerate(grads_budget)]


            apply_gradient_op_degradation = opt_degradation.apply_gradients([(accum_vars_degradtion[i].value(), gv[1]) for i, gv in enumerate(grads_degradation)], global_step=global_step)
            apply_gradient_op_utility = opt_utility.apply_gradients([(accum_vars_utility[i].value(), gv[1]) for i, gv in enumerate(grads_utility)], global_step=global_step)
            apply_gradient_op_budget = opt_budget.apply_gradients([(accum_vars_budget[i].value(), gv[1]) for i, gv in enumerate(grads_budget)], global_step=global_step)


            train_video_files = [os.path.join(FLAGS.train_videos_files_dir, f) for f in
                           os.listdir(FLAGS.train_videos_files_dir) if f.endswith('.tfrecords')]
            val_video_files = [os.path.join(FLAGS.val_videos_files_dir, f) for f in
                         os.listdir(FLAGS.val_videos_files_dir) if f.endswith('.tfrecords')]
            print(train_video_files)
            print(val_video_files)

            train_image_files = [os.path.join(FLAGS.train_images_files_dir, f) for f in
                           os.listdir(FLAGS.train_images_files_dir) if f.endswith('.tfrecords')]
            print(train_image_files)
            val_image_files = [os.path.join(FLAGS.val_images_files_dir, f) for f in
                           os.listdir(FLAGS.val_images_files_dir) if f.endswith('.tfrecords')]
            print(val_image_files)
            test_image_files = [os.path.join(FLAGS.test_images_files_dir, f) for f in
                           os.listdir(FLAGS.test_images_files_dir) if f.endswith('.tfrecords')]
            print(test_image_files)

            tr_videos_op, tr_videos_labels_op = inputs_videos(filenames = train_video_files,
                                                 batch_size=FLAGS.video_batch_size * FLAGS.gpu_num,
                                                 num_epochs=None,
                                                 num_threads=FLAGS.num_threads,
                                                 num_examples_per_epoch=FLAGS.num_examples_per_epoch,
                                                 shuffle=True)
            val_videos_op, val_videos_labels_op = inputs_videos(filenames = val_video_files,
                                                   batch_size=FLAGS.video_batch_size * FLAGS.gpu_num,
                                                   num_epochs=None,
                                                   num_threads=FLAGS.num_threads,
                                                   num_examples_per_epoch=FLAGS.num_examples_per_epoch,
                                                   shuffle=True)
            tr_images_op, tr_images_labels_op = inputs_images(filenames = train_image_files + val_image_files,
                                                   batch_size=FLAGS.image_batch_size * FLAGS.gpu_num,
                                                   num_epochs=None,
                                                   num_threads=FLAGS.num_threads,
                                                   num_examples_per_epoch=FLAGS.num_examples_per_epoch,
                                                   shuffle=True)
            val_images_op, val_images_labels_op = inputs_images(filenames = test_image_files,
                                                   batch_size=FLAGS.image_batch_size * FLAGS.gpu_num,
                                                   num_epochs=None,
                                                   num_threads=FLAGS.num_threads,
                                                   num_examples_per_epoch=FLAGS.num_examples_per_epoch,
                                                   shuffle=True)

            init_op = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())
            sess.run(init_op)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            def restore_model_ckpt(ckpt_dir, varlist, modulename):
                ckpt = tf.train.get_checkpoint_state(checkpoint_dir=ckpt_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    print("##########################{}##########################".format(modulename))
                    print(varlist)
                    saver = tf.train.Saver(varlist)
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    print(
                        'Session restored from pretrained model at {}!'.format(ckpt.model_checkpoint_path))
                else:
                    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), ckpt_dir)

            def restore_model_pretrained_C3D():
                if os.path.isfile(FLAGS.pretrained_C3D):
                    varlist = [v for v in tf.trainable_variables() if
                                       any(x in v.name for x in ["UtilityModule"])]
                    varlist = [v for v in varlist if not any(x in v.name.split('/')[1] for x in ["out", "d2"])]
                    vardict = {v.name[:-2].replace('UtilityModule', 'var_name'):v for v in varlist}
                    for key, value in vardict.items():
                        print(key)
                    saver = tf.train.Saver(vardict)
                    saver.restore(sess, FLAGS.pretrained_C3D)
                    print('#############################Session restored from pretrained model at {}!#############################'.format(FLAGS.pretrained_C3D))
                else:
                    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), FLAGS.pretrained_C3D)

            # Create a saver for writing training checkpoints.
            if use_pretrained_model:
                restore_model_ckpt(FLAGS.degradation_models, varlist_degrad, "DegradationModule")
                restore_model_pretrained_C3D()
                restore_model_ckpt(FLAGS.budget_multi_models, varlist_budget, "BudgetModule")
            elif use_whole_pretrained_model:
                saver = tf.train.Saver(varlist_utility + varlist_degrad)
                ckpt = tf.train.get_checkpoint_state(checkpoint_dir=FLAGS.whole_pretrained_checkpoint_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    print('Session restored from trained model at {}!'.format(ckpt.model_checkpoint_path))
                else:
                    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), FLAGS.whole_pretrained_checkpoint_dir)
                saver = tf.train.Saver(varlist_budget)
                ckpt = tf.train.get_checkpoint_state(checkpoint_dir=FLAGS.budget_multi_models)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    print('Session restored from trained model at {}!'.format(ckpt.model_checkpoint_path))
                else:
                    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), FLAGS.budget_multi_models)
            else:
                saver = tf.train.Saver()
                ckpt = tf.train.get_checkpoint_state(checkpoint_dir=FLAGS.checkpoint_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    print('Session restored from trained model at {}!'.format(ckpt.model_checkpoint_path))
                else:
                    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), FLAGS.checkpoint_dir)

            gvar_list = tf.global_variables()
            bn_moving_vars = [g for g in gvar_list if 'moving_mean' in g.name]
            bn_moving_vars += [g for g in gvar_list if 'moving_variance' in g.name]
            #saver = tf.train.Saver(tf.trainable_variables() + bn_moving_vars)
            whole_model_saver = tf.train.Saver(max_to_keep=5)
            if not os.path.exists(FLAGS.summary_dir):
                os.makedirs(FLAGS.summary_dir)
            loss_summary_file = open(FLAGS.summary_dir+'loss_summary.txt', 'w')
            train_summary_file = open(FLAGS.summary_dir+'train_summary.txt', 'w')
            test_summary_file = open(FLAGS.summary_dir+'test_summary.txt', 'w')

            privacy_uniform_labels = np.full((FLAGS.video_batch_size * FLAGS.gpu_num, FLAGS.num_classes_budget),
                                              1, dtype=np.float32)

            ckpt_saver = tf.train.Saver()
            for step in xrange(FLAGS.max_steps):
                # if step == 0 or (FLAGS.use_resampling and step % FLAGS.resample_step == 0):
                #     saver = tf.train.Saver(varlist_budget)
                #     ckpt = tf.train.get_checkpoint_state(checkpoint_dir=FLAGS.budget_multi_models)
                #     if ckpt and ckpt.model_checkpoint_path:
                #         saver.restore(sess, ckpt.model_checkpoint_path)
                #         print('Session restored from trained model at {}!'.format(ckpt.model_checkpoint_path))
                #     else:
                #         raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), FLAGS.budget_multi_models)
                #     for step in xrange(FLAGS.pretraining_steps_budget):
                #         start_time = time.time()
                #         sess.run(zero_ops_budget)
                #         loss_budget_lst = []
                #         for _ in itertools.repeat(None, 12):
                #             tr_images, tr_images_labels = sess.run([tr_images_op, tr_images_labels_op])
                #             _, loss_budget = sess.run([accum_ops_budget, loss_budget_op],
                #                                   feed_dict={images_placeholder: tr_images,
                #                                              budget_images_labels_placeholder: tr_images_labels,
                #                                              dropout_placeholder: 1.0,
                #                                              isTraining_placeholder: True})
                #             loss_budget_lst.append(loss_budget)
                #         sess.run([apply_gradient_op_budget])
                #         loss_summary = 'Budget Module Pretraining, Step: {:4d}, time: {:.4f}, budget loss: {:.8f}'.format(step,
                #                                                    time.time() - start_time, np.mean(loss_budget_lst))
                #         print(loss_summary)
                #
                #         if step % FLAGS.val_step == 0:
                #             loss_budget_lst = []
                #             pred_probs_lst = []
                #             gt_lst = []
                #             for _ in itertools.repeat(None, 30):
                #                 val_images, val_images_labels = sess.run([val_images_op, val_images_labels_op])
                #                 gt_lst.append(val_images_labels)
                #                 logits_budget, loss_budget = sess.run([logits_budget_images_op, loss_budget_op],
                #                                                   feed_dict={images_placeholder: val_images,
                #                                                              budget_images_labels_placeholder: val_images_labels,
                #                                                              dropout_placeholder: 1.0,
                #                                                              isTraining_placeholder: True})
                #                 loss_budget_lst.append(loss_budget)
                #                 pred_probs_lst.append(logits_budget)
                #
                #             pred_probs_mat = np.concatenate(pred_probs_lst, axis=0)
                #             gt_mat = np.concatenate(gt_lst, axis=0)
                #             n_examples, n_labels = gt_mat.shape
                #             print('# Examples = ', n_examples)
                #             print('# Labels = ', n_labels)
                #             print('Macro MAP = {:.2f}'.format(100 * average_precision_score(gt_mat, pred_probs_mat, average='macro')))

                start_time = time.time()
                loss_filter_lst, loss_utility_lst = [], []
                sess.run(zero_ops_degradation)
                for _ in itertools.repeat(None, FLAGS.n_minibatches):
                    tr_videos, tr_videos_labels = sess.run(
                                    [tr_videos_op, tr_videos_labels_op])
                    _, argmax_cent, loss_filter, loss_utility = sess.run([accum_ops_degradation, argmax_cent_op, loss_filter_op, loss_utility_op],
                                                    feed_dict={videos_placeholder: tr_videos,
                                                               utility_labels_placeholder: tr_videos_labels,
                                                               budget_videos_labels_placeholder: privacy_uniform_labels,
                                                               dropout_placeholder: 1.0,
                                                               isTraining_placeholder: True})
                    print(argmax_cent)
                    loss_filter_lst.append(loss_filter)
                    loss_utility_lst.append(loss_utility)
                sess.run(apply_gradient_op_degradation)
                duration = time.time() - start_time
                loss_summary = 'Alternating Training (Degradation), Step: {:4d}, time: {:.4f}, filter loss: {:.8f}, utility loss: {:.8f}'.format(step,
                                                        duration, np.mean(loss_filter_lst), np.mean(loss_utility_lst))

                print(loss_summary)
                loss_summary_file.write(loss_summary + '\n')

                # while True:
                #     start_time = time.time()
                #     acc_util_lst, loss_filter_lst, loss_utility_lst = [], [], []
                #     for _ in itertools.repeat(None, FLAGS.n_minibatches):
                #         val_videos, val_videos_labels = sess.run(
                #             [val_videos_op, val_videos_labels_op])
                #         acc_util, loss_filter, loss_utility = sess.run(
                #             [accuracy_util, loss_filter_op, loss_utility_op],
                #             feed_dict={videos_placeholder: val_videos,
                #                        utility_labels_placeholder: val_videos_labels,
                #                        budget_videos_labels_placeholder: privacy_uniform_labels,
                #                        dropout_placeholder: 1.0,
                #                        isTraining_placeholder: True,
                #                        })
                #         acc_util_lst.append(acc_util)
                #         loss_filter_lst.append(loss_filter)
                #         loss_utility_lst.append(loss_utility)
                #     val_summary = "Monitoring (Utility), Step: {:4d}, time: {:.4f}, filter loss: {:.8f}, utility loss: {:.8f}, validation utility accuracy: {:.5f},\n".format(
                #         step,
                #         time.time() - start_time, np.mean(loss_filter_lst),
                #         np.mean(loss_utility_lst), np.mean(acc_util_lst))
                #     print(val_summary)
                #
                #     if np.mean(acc_util_lst) >= FLAGS.highest_util_acc_val:
                #         break
                #     start_time = time.time()
                #     sess.run(zero_ops_utility)
                #     acc_util_lst, acc_budget_lst, loss_filter_lst, loss_utility_lst, loss_budget_lst = [], [], [], [], []
                #     for _ in itertools.repeat(None, FLAGS.n_minibatches):
                #         tr_videos, tr_videos_labels = sess.run(
                #             [tr_videos_op, tr_videos_labels_op])
                #         _, acc_util, loss_filter, loss_utility = sess.run(
                #             [accum_ops_utility, accuracy_util, loss_filter_op, loss_utility_op],
                #             feed_dict={videos_placeholder: tr_videos,
                #                        utility_labels_placeholder: tr_videos_labels,
                #                        budget_videos_labels_placeholder: privacy_uniform_labels,
                #                        dropout_placeholder: 1.0,
                #                        isTraining_placeholder: True,
                #                        })
                #         acc_util_lst.append(acc_util)
                #         loss_filter_lst.append(loss_filter)
                #         loss_utility_lst.append(loss_utility)
                #     sess.run([apply_gradient_op_utility])
                #     loss_summary = 'Alternating Training (Utility), Step: {:4d}, time: {:.4f}, filter loss: {:.8f}, utility loss: {:.8f}'.format(
                #         step,
                #         time.time() - start_time, np.mean(loss_filter_lst),
                #         np.mean(loss_utility_lst))
                #     print(loss_summary)

                for _ in itertools.repeat(None, 5):
                    start_time = time.time()
                    sess.run(zero_ops_budget)
                    loss_budget_lst = []
                    for _ in itertools.repeat(None, FLAGS.n_minibatches):
                        tr_images, tr_images_labels = sess.run(
                            [tr_images_op, tr_images_labels_op])

                        _, loss_budget = sess.run(
                            [accum_ops_budget, loss_budget_op],
                            feed_dict={images_placeholder: tr_images,
                                       budget_images_labels_placeholder: tr_images_labels,
                                       dropout_placeholder: 1.0,
                                       isTraining_placeholder: True,})
                        loss_budget_lst.append(loss_budget)
                    sess.run([apply_gradient_op_budget])
                    loss_summary = 'Alternating Training (Budget), Step: {:4d}, time: {:.4f}, budget loss: {:.8f}'.format(step,
                                                                                                                time.time() - start_time,
                                                                                                                np.mean(
                                                                                                                    loss_budget_lst))
                    print(loss_summary)

                if step % FLAGS.val_step == 0:
                    start_time = time.time()
                    acc_util_train_lst, loss_filter_train_lst, loss_utility_train_lst = [], [], []
                    acc_util_val_lst, loss_filter_val_lst, loss_utility_val_lst = [], [], []
                    for _ in itertools.repeat(None, 60):
                        tr_videos, tr_videos_labels = sess.run(
                            [tr_videos_op, tr_videos_labels_op])
                        acc_util, loss_filter, loss_utility = sess.run(
                            [accuracy_util, loss_filter_op, loss_utility_op],
                            feed_dict={videos_placeholder: tr_videos,
                                       utility_labels_placeholder: tr_videos_labels,
                                       budget_videos_labels_placeholder: privacy_uniform_labels,
                                       dropout_placeholder: 1.0,
                                       isTraining_placeholder: True,
                                       })
                        acc_util_train_lst.append(acc_util)
                        loss_filter_train_lst.append(loss_filter)
                        loss_utility_train_lst.append(loss_utility)

                    train_summary = "Step: {:4d}, time: {:.4f}, filter loss: {:.8f}, utility loss: {:.8f}, training utility accuracy: {:.5f}".format(
                        step,
                        time.time() - start_time, np.mean(loss_filter_train_lst),
                        np.mean(loss_utility_train_lst), np.mean(acc_util_train_lst))
                    print(train_summary)
                    train_summary_file.write(train_summary + '\n')

                    for _ in itertools.repeat(None, 60):
                        val_videos, val_videos_labels = sess.run(
                            [val_videos_op, val_videos_labels_op])
                        acc_util, loss_filter_value, loss_utility_value = sess.run(
                            [accuracy_util, loss_filter_op, loss_utility_op],
                            feed_dict={videos_placeholder: val_videos,
                                       utility_labels_placeholder: val_videos_labels,
                                       budget_videos_labels_placeholder: privacy_uniform_labels,
                                       dropout_placeholder: 1.0,
                                       isTraining_placeholder: True,
                                       })
                        acc_util_val_lst.append(acc_util)
                        loss_filter_val_lst.append(loss_filter_value)
                        loss_utility_val_lst.append(loss_utility_value)

                    test_summary = "Step: {:4d}, time: {:.4f}, filter loss: {:.8f}, utility loss: {:.8f}, validation utility accuracy: {:.5f}".format(
                        step,
                        time.time() - start_time, np.mean(loss_filter_val_lst),
                        np.mean(loss_utility_val_lst), np.mean(acc_util_val_lst))
                    print(test_summary)
                    test_summary_file.write(test_summary + '\n')

                    start_time = time.time()
                    loss_budget_lst = []
                    pred_probs_lst = []
                    gt_lst = []
                    for _ in itertools.repeat(None, 60):
                        val_images, val_images_labels = sess.run(
                                [val_images_op, val_images_labels_op])
                        gt_lst.append(val_images_labels)
                        logits_budget, loss_budget = sess.run([logits_budget_images_op, loss_budget_op],
                                feed_dict={images_placeholder: val_images,
                                           budget_images_labels_placeholder: val_images_labels,
                                           dropout_placeholder: 1.0,
                                           isTraining_placeholder: True})
                        loss_budget_lst.append(loss_budget)
                        pred_probs_lst.append(logits_budget)


                    pred_probs_mat = np.concatenate(pred_probs_lst, axis=0)
                    gt_mat = np.concatenate(gt_lst, axis=0)
                    n_examples, n_labels = gt_mat.shape
                    print('# Examples = ', n_examples)
                    print('# Labels = ', n_labels)
                    test_summary = "Step: {:4d}, time: {:.4f}, Macro MAP = {:.2f}".format(
                        step,
                        time.time() - start_time,
                        100 * average_precision_score(gt_mat, pred_probs_mat, average='macro'))
                    print(test_summary)
                    test_summary_file.write(test_summary + '\n')

                if step % FLAGS.save_step == 0 or (step + 1) == FLAGS.max_steps:
                    checkpoint_path = os.path.join(FLAGS.checkpoint_dir, 'model.ckpt')
                    ckpt_saver.save(sess, checkpoint_path, global_step=step)

            loss_summary_file.close()
            train_summary_file.close()
            test_summary_file.close()
            coord.request_stop()
            coord.join(threads)

        print("done")

def run_testing_multi_model_residual_vispr():
    # Create model directory
    #if not os.path.exists(FLAGS.checkpoint_dir):
    #    os.makedirs(FLAGS.checkpoint_dir)

    dir_path = FLAGS.checkpoint_dir
    ckpt_files = [".".join(f.split(".")[:-1]) for f in os.listdir(dir_path) if os.path.isfile(
        os.path.join(dir_path, f)) and '.data' in f]

    for ckpt_file in ckpt_files:
        for is_training in [True, False]:
            config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
            config.gpu_options.allow_growth = True
            tf.reset_default_graph()
            with tf.Graph().as_default():
                with tf.Session(config=config) as sess:
                    images_placeholder = tf.placeholder(tf.float32, shape=(FLAGS.image_batch_size * FLAGS.gpu_num, None, None, FLAGS.nchannel))
                    labels_placeholder = tf.placeholder(tf.float32, shape=(FLAGS.image_batch_size * FLAGS.gpu_num, FLAGS.num_classes_budget))
                    dropout_placeholder = tf.placeholder(tf.float32)
                    isTraining_placeholder = tf.placeholder(tf.bool)

                    from collections import defaultdict
                    logits_budget_images_lst_dct = defaultdict(list)
                    loss_budget_images_lst_dct = defaultdict(list)

                    logits_budget_images_lst = []
                    loss_budget_images_lst = []

                    budgetNet_model_name_dict = {4:['resnet_v1_50', 'resnet_v2_50', 'mobilenet_v1', 'mobilenet_v1_075'],
                                             3:['resnet_v1_50', 'resnet_v2_50', 'mobilenet_v1'],
                                             2:['resnet_v1_50', 'resnet_v2_50'],
                                             1:['resnet_v1_50']}
                    budgetNet_model_name_lst = budgetNet_model_name_dict[FLAGS.NBudget]

                    budgetNet_dict = {}
                    for model_name in budgetNet_model_name_lst:
                        budgetNet_dict[model_name] = nets_factory.get_network_fn(model_name,
                                                                             num_classes=FLAGS.num_classes_budget,
                                                                             weight_decay=FLAGS.weight_decay,
                                                                             is_training=isTraining_placeholder)
                    with tf.variable_scope(tf.get_variable_scope()) as scope:
                        for gpu_index in range(0, FLAGS.gpu_num):
                            with tf.device('/gpu:%d' % gpu_index):
                                print('/gpu:%d' % gpu_index)
                                with tf.name_scope('%s_%d' % ('gpu', gpu_index)) as scope:
                                    degrad_images = residualNet(images_placeholder[gpu_index * FLAGS.image_batch_size:(gpu_index + 1) * FLAGS.image_batch_size], training=False)
                                    loss_budget_images = 0.0
                                    logits_budget_images = tf.zeros([FLAGS.image_batch_size, FLAGS.num_classes_budget])
                                    for model_name in budgetNet_model_name_lst:
                                        print(model_name)
                                        print(tf.trainable_variables())
                                        with tf.variable_scope(tf.get_variable_scope(), reuse=False):
                                            logits, _ = budgetNet_dict[model_name](degrad_images)
                                            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                                                labels=labels_placeholder[gpu_index * FLAGS.image_batch_size:(gpu_index + 1) * FLAGS.image_batch_size,:]))
                                        logits_budget_images += logits
                                        logits_budget_images_lst_dct[model_name].append(logits)
                                        loss_budget_images_lst_dct[model_name].append(loss)
                                        loss_budget_images += loss
                                    loss_budget_images_lst.append(loss_budget_images)
                                    logits_budget_images = tf.divide(logits_budget_images, 4.0, 'LogitsBudgetMean')
                                    logits_budget_images_lst.append(logits_budget_images)

                                    tf.get_variable_scope().reuse_variables()

                    loss_budget_op = tf.reduce_mean(loss_budget_images_lst, name='softmax')

                    logits_budget_images_op = tf.concat(logits_budget_images_lst, axis=0)

                    logits_budget_images_op_lst = []
                    for model_name in budgetNet_model_name_lst:
                        logits_budget_images_op_lst.append(tf.concat(logits_budget_images_lst_dct[model_name], axis=0))

                    train_image_files = [os.path.join(FLAGS.train_images_files_dir, f) for f in
                           os.listdir(FLAGS.train_images_files_dir) if f.endswith('.tfrecords')]
                    print(train_image_files)
                    val_image_files = [os.path.join(FLAGS.val_images_files_dir, f) for f in
                           os.listdir(FLAGS.val_images_files_dir) if f.endswith('.tfrecords')]
                    print(val_image_files)
                    test_image_files = [os.path.join(FLAGS.test_images_files_dir, f) for f in
                           os.listdir(FLAGS.test_images_files_dir) if f.endswith('.tfrecords')]
                    print(test_image_files)

                    if is_training:
                        images_op, images_labels_op = inputs_images(filenames=train_image_files,
                                                                               batch_size=FLAGS.image_batch_size * FLAGS.gpu_num,
                                                                               num_epochs=1,
                                                                               num_threads=FLAGS.num_threads,
                                                                               num_examples_per_epoch=FLAGS.num_examples_per_epoch,
                                                                               shuffle=False)
                    else:
                        images_op, images_labels_op = inputs_images(filenames = test_image_files,
                                                   batch_size=FLAGS.image_batch_size * FLAGS.gpu_num,
                                                   num_epochs=1,
                                                   num_threads=FLAGS.num_threads,
                                                   num_examples_per_epoch=FLAGS.num_examples_per_epoch,
                                                   shuffle=False)

                    init_op = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())
                    sess.run(init_op)

                    coord = tf.train.Coordinator()
                    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

                    saver = tf.train.Saver(tf.trainable_variables())

                    saver.restore(sess, dir_path + ckpt_file)
                    print('Session restored from trained model at {}!'.format(dir_path + ckpt_file))

                    budgetNet_model_name_lst += ['ensemble']
                    loss_budget_lst = []
                    pred_probs_lst_lst = [[] for _ in xrange(len(budgetNet_model_name_lst))]
                    gt_lst = []
                    try:
                        while not coord.should_stop():
                            images, labels = sess.run(
                                    [images_op, images_labels_op])
                            gt_lst.append(labels)
                            value_lst = sess.run([loss_budget_op, logits_budget_images_op] + logits_budget_images_op_lst,
                                        feed_dict={images_placeholder: images,
                                                   labels_placeholder: labels,
                                                   dropout_placeholder: 1.0,
                                                   isTraining_placeholder: True})
                            print(labels.shape)
                            loss_budget_lst.append(value_lst[0])
                            for i in xrange(len(budgetNet_model_name_lst)):
                                pred_probs_lst_lst[i].append(value_lst[i+1])
                    except tf.errors.OutOfRangeError:
                        print('Done testing on all the examples')
                    finally:
                        coord.request_stop()

                    gt_mat = np.concatenate(gt_lst, axis=0)
                    n_examples, n_labels = gt_mat.shape
                    for i in xrange(len(budgetNet_model_name_lst)):
                        save_dir = os.path.join(FLAGS.checkpoint_dir, ckpt_file.split('.')[-1])
                        if not os.path.exists(save_dir):
                            os.makedirs(save_dir)
                        isTraining = lambda bool: "training" if bool else "validation"
                        with open(os.path.join(save_dir, '{}_class_scores_{}.txt'.format(budgetNet_model_name_lst[i], isTraining(is_training))), 'w') as wf:
                            pred_probs_mat = np.concatenate(pred_probs_lst_lst[i], axis=0)
                            wf.write('# Examples = {}\n'.format(n_examples))
                            wf.write('# Labels = {}\n'.format(n_labels))
                            wf.write('Average Loss = {}\n'.format(np.mean(loss_budget_lst)))
                            wf.write("Macro MAP = {:.2f}\n".format(100 * average_precision_score(gt_mat, pred_probs_mat, average='macro')))
                            cmap_stats = average_precision_score(gt_mat, pred_probs_mat, average=None)
                            attr_id_to_name, attr_id_to_idx = load_attributes()
                            idx_to_attr_id = {v: k for k, v in attr_id_to_idx.items()}
                            wf.write('\t'.join(['attribute_id', 'attribute_name', 'num_occurrences', 'ap']) + '\n')
                            for idx in range(n_labels):
                                attr_id = idx_to_attr_id[idx]
                                attr_name = attr_id_to_name[attr_id]
                                attr_occurrences = np.sum(gt_mat, axis=0)[idx]
                                ap = cmap_stats[idx]
                                wf.write('{}\t{}\t{}\t{}\n'.format(attr_id, attr_name, attr_occurrences, ap * 100.0))

                    coord.join(threads)
                    sess.close()

def run_testing_multi_model_residual_ucf():
    dir_path = FLAGS.checkpoint_dir
    ckpt_files = [".".join(f.split(".")[:-1]) for f in os.listdir(dir_path) if os.path.isfile(
                                        os.path.join(dir_path, f)) and '.data' in f]

    for ckpt_file in ckpt_files:
        for is_training in [True, False]:
            config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
            config.gpu_options.allow_growth = True
            with tf.Graph().as_default():
                with tf.Session(config=config) as sess:
                    videos_placeholder = tf.placeholder(tf.float32, shape=(FLAGS.video_batch_size * FLAGS.gpu_num, FLAGS.depth, None, None, FLAGS.nchannel))
                    utility_labels_placeholder = tf.placeholder(tf.int64, shape=(FLAGS.video_batch_size * FLAGS.gpu_num))
                    dropout_placeholder = tf.placeholder(tf.float32)
                    logits_utility_lst = []
                    loss_utility_lst = []
                    with tf.variable_scope(tf.get_variable_scope()) as scope:
                        for gpu_index in range(0, FLAGS.gpu_num):
                            with tf.device('/gpu:%d' % gpu_index):
                                print('/gpu:%d' % gpu_index)
                                with tf.name_scope('%s_%d' % ('gpu', gpu_index)) as scope:
                                    videos = tf.reshape(videos_placeholder[gpu_index * FLAGS.video_batch_size:(gpu_index + 1) * FLAGS.video_batch_size],
                                                [FLAGS.video_batch_size * FLAGS.depth, FLAGS.crop_height, FLAGS.crop_width, FLAGS.nchannel])
                                    videos = residualNet(videos, training=False)
                                    videos_utility = tf.reshape(videos,
                                                [FLAGS.video_batch_size, FLAGS.depth, FLAGS.crop_height, FLAGS.crop_width, FLAGS.nchannel])
                                    logits_utility = C3DNet(videos_utility, dropout_placeholder)
                                    logits_utility_lst.append(logits_utility)
                                    loss_utility = tower_loss_xentropy_sparse(
                                                        scope,
                                                        logits_utility,
                                                        utility_labels_placeholder[gpu_index * FLAGS.video_batch_size:(gpu_index + 1) * FLAGS.video_batch_size],
                                                        use_weight_decay = True
                                    )
                                    loss_utility_lst.append(loss_utility)
                                    tf.get_variable_scope().reuse_variables()

                    logits_utility_op = tf.concat(logits_utility_lst, 0)
                    right_count_utility_op = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(tf.nn.softmax(logits_utility_op), axis=1), utility_labels_placeholder), tf.int32))

                    train_video_files = [os.path.join(FLAGS.train_videos_files_dir, f) for f in os.listdir(FLAGS.train_videos_files_dir) if f.endswith('.tfrecords')]
                    val_video_files = [os.path.join(FLAGS.val_videos_files_dir, f) for f in os.listdir(FLAGS.val_videos_files_dir) if f.endswith('.tfrecords')]
                    print(train_video_files)
                    print(val_video_files)

                    if is_training:
                        videos_op, videos_labels_op = inputs_videos(filenames=train_video_files,
                                                                               batch_size=FLAGS.video_batch_size * FLAGS.gpu_num,
                                                                               num_epochs=1,
                                                                               num_threads=FLAGS.num_threads,
                                                                               num_examples_per_epoch=FLAGS.num_examples_per_epoch,
                                                                               shuffle=False)
                    else:
                        videos_op, videos_labels_op = inputs_videos(filenames = val_video_files,
                                                                       batch_size=FLAGS.video_batch_size * FLAGS.gpu_num,
                                                                       num_epochs=1,
                                                                       num_threads=FLAGS.num_threads,
                                                                       num_examples_per_epoch=FLAGS.num_examples_per_epoch,
                                                                       shuffle=False)

                    init_op = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())
                    sess.run(init_op)

                    coord = tf.train.Coordinator()
                    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

                    # Create a saver for writing training checkpoints.
                    saver = tf.train.Saver(tf.trainable_variables())
                    saver.restore(sess, dir_path + ckpt_file)
                    print('Session restored from trained model at {}!'.format(dir_path + ckpt_file))

                    total_v_utility = 0.0
                    test_correct_num_utility = 0.0
                    try:
                        while not coord.should_stop():
                            videos, videos_labels = sess.run([videos_op, videos_labels_op])
                            feed = {videos_placeholder: videos, utility_labels_placeholder: videos_labels, dropout_placeholder: 1.0}
                            right_utility = sess.run(right_count_utility_op, feed_dict=feed)
                            print(total_v_utility)
                            test_correct_num_utility += right_utility
                            total_v_utility += videos_labels.shape[0]

                    except tf.errors.OutOfRangeError:
                        print('Done testing on all the examples')
                    finally:
                        coord.request_stop()

                    save_dir = os.path.join(FLAGS.checkpoint_dir, ckpt_file.split('.')[-1])
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    isTraining = lambda bool: "training" if bool else "validation"
                    with open(os.path.join(save_dir, 'C3D_accuracy_{}.txt'.format(isTraining(is_training))), 'w') as wf:
                        wf.write('Utility test acc: {}'.format(test_correct_num_utility / total_v_utility))
                        wf.write('Utility test_correct_num: {}'.format(test_correct_num_utility))
                        wf.write('Utility total_v: {}'.format(total_v_utility))

                    coord.join(threads)
                    sess.close()

def run_testing_vispr():
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True
    with tf.Graph().as_default():
        with tf.Session(config=config) as sess:
            images_placeholder = tf.placeholder(tf.float32, shape=(FLAGS.image_batch_size * FLAGS.gpu_num, None, None, FLAGS.nchannel))
            budget_labels_placeholder = tf.placeholder(tf.float32, shape=(FLAGS.image_batch_size * FLAGS.gpu_num, FLAGS.num_classes_budget))
            dropout_placeholder = tf.placeholder(tf.float32)
            isTraining_placeholder = tf.placeholder(tf.bool)

            logits_budget_images_lst = []

            loss_budget_images_lst = []

            budgetNet = nets_factory.get_network_fn(FLAGS.model_name,
                                                    num_classes=FLAGS.num_classes_budget,
                                                    weight_decay=FLAGS.weight_decay,
                                                    is_training=isTraining_placeholder)
            with tf.variable_scope(tf.get_variable_scope()) as scope:
                for gpu_index in range(0, FLAGS.gpu_num):
                    with tf.device('/gpu:%d' % gpu_index):
                        print('/gpu:%d' % gpu_index)
                        with tf.name_scope('%s_%d' % ('gpu', gpu_index)) as scope:
                            LR_images = residualNet(images_placeholder[gpu_index * FLAGS.image_batch_size:(gpu_index + 1) * FLAGS.image_batch_size], training=False)
                            logits_budget_images, _ = budgetNet(LR_images)
                            logits_budget_images_lst.append(logits_budget_images)
                            loss_budget_images = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = logits_budget_images,
                                                labels = budget_labels_placeholder[gpu_index * FLAGS.image_batch_size:(gpu_index + 1) * FLAGS.image_batch_size, :]))
                            loss_budget_images_lst.append(loss_budget_images)

                            tf.get_variable_scope().reuse_variables()

            loss_budget_op = tf.reduce_mean(loss_budget_images_lst, name='softmax')
            logits_budget_images_op = tf.concat(logits_budget_images_lst, 0)

            train_image_files = [os.path.join(FLAGS.train_images_files_dir, f) for f in
                           os.listdir(FLAGS.train_images_files_dir) if f.endswith('.tfrecords')]
            print(train_image_files)
            val_image_files = [os.path.join(FLAGS.val_images_files_dir, f) for f in
                           os.listdir(FLAGS.val_images_files_dir) if f.endswith('.tfrecords')]
            print(val_image_files)
            test_image_files = [os.path.join(FLAGS.test_images_files_dir, f) for f in
                           os.listdir(FLAGS.test_images_files_dir) if f.endswith('.tfrecords')]
            print(test_image_files)

            images_op, images_labels_op = inputs_images(filenames = test_image_files,
                                                   batch_size=FLAGS.image_batch_size * FLAGS.gpu_num,
                                                   num_epochs=1,
                                                   num_threads=FLAGS.num_threads,
                                                   num_examples_per_epoch=FLAGS.num_examples_per_epoch,
                                                   shuffle=True)

            init_op = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())
            sess.run(init_op)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)



            # Create a saver for writing training checkpoints.
            saver = tf.train.Saver(tf.trainable_variables())
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir=FLAGS.checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Session restored from trained model at {}!'.format(ckpt.model_checkpoint_path))
            else:
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), FLAGS.checkpoint_dir)

            loss_budget_lst = []
            pred_probs_lst = []
            gt_lst = []
            try:
                while not coord.should_stop():
                    images, images_labels = sess.run(
                                [images_op, images_labels_op])
                    gt_lst.append(images_labels)
                    logits_budget, loss_budget = sess.run([logits_budget_images_op, loss_budget_op],
                                feed_dict={images_placeholder: images,
                                           budget_labels_placeholder: images_labels,
                                           dropout_placeholder: 1.0,
                                           isTraining_placeholder: True})
                    print(images_labels.shape)
                    print(logits_budget.shape)
                    loss_budget_lst.append(loss_budget)
                    pred_probs_lst.append(logits_budget)
            except tf.errors.OutOfRangeError:
                print('Done testing on all the examples')
            finally:
                coord.request_stop()

            pred_probs_mat = np.concatenate(pred_probs_lst, axis=0)
            gt_mat = np.concatenate(gt_lst, axis=0)
            n_examples, n_labels = gt_mat.shape
            print('# Examples = ', n_examples)
            print('# Labels = ', n_labels)
            print('Average Loss = ', np.mean(loss_budget_lst))
            print("Macro MAP = {:.2f}".format(100 * average_precision_score(gt_mat, pred_probs_mat, average='macro')))

            coord.join(threads)
            sess.close()

def run_testing_ucf101():
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True
    with tf.Graph().as_default():
        with tf.Session(config=config) as sess:
            videos_placeholder = tf.placeholder(tf.float32, shape=(FLAGS.video_batch_size * FLAGS.gpu_num, FLAGS.depth, None, None, FLAGS.nchannel))
            utility_labels_placeholder = tf.placeholder(tf.int64, shape=(FLAGS.video_batch_size * FLAGS.gpu_num))
            dropout_placeholder = tf.placeholder(tf.float32)

            logits_utility_lst = []

            loss_utility_lst = []

            with tf.variable_scope(tf.get_variable_scope()) as scope:
                for gpu_index in range(0, FLAGS.gpu_num):
                    with tf.device('/gpu:%d' % gpu_index):
                        print('/gpu:%d' % gpu_index)
                        with tf.name_scope('%s_%d' % ('gpu', gpu_index)) as scope:
                            videos = tf.reshape(videos_placeholder[gpu_index * FLAGS.video_batch_size:(gpu_index + 1) * FLAGS.video_batch_size],
                                                [FLAGS.video_batch_size * FLAGS.depth, FLAGS.crop_height, FLAGS.crop_width, FLAGS.nchannel])
                            videos = residualNet(videos, training=False)
                            videos_utility = tf.reshape(videos,
                                                [FLAGS.video_batch_size, FLAGS.depth, FLAGS.crop_height, FLAGS.crop_width, FLAGS.nchannel])
                            logits_utility = C3DNet(videos_utility, dropout_placeholder)
                            logits_utility_lst.append(logits_utility)
                            loss_utility = tower_loss_xentropy_sparse(
                                scope,
                                logits_utility,
                                utility_labels_placeholder[gpu_index * FLAGS.video_batch_size:(gpu_index + 1) * FLAGS.video_batch_size],
                                use_weight_decay = True
                            )
                            loss_utility_lst.append(loss_utility)

                            tf.get_variable_scope().reuse_variables()

            logits_utility_op = tf.concat(logits_utility_lst, 0)
            right_count_utility_op = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(tf.nn.softmax(logits_utility_op), axis=1), utility_labels_placeholder), tf.int32))

            train_video_files = [os.path.join(FLAGS.train_videos_files_dir, f) for f in
                           os.listdir(FLAGS.train_videos_files_dir) if f.endswith('.tfrecords')]
            val_video_files = [os.path.join(FLAGS.val_videos_files_dir, f) for f in
                         os.listdir(FLAGS.val_videos_files_dir) if f.endswith('.tfrecords')]
            print(train_video_files)
            print(val_video_files)

            videos_op, videos_labels_op = inputs_videos(filenames = val_video_files,
                                                 batch_size=FLAGS.video_batch_size * FLAGS.gpu_num,
                                                 num_epochs=1,
                                                 num_threads=FLAGS.num_threads,
                                                 num_examples_per_epoch=FLAGS.num_examples_per_epoch,
                                                 shuffle=True)

            init_op = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())
            sess.run(init_op)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            # Create a saver for writing training checkpoints.
            saver = tf.train.Saver(tf.trainable_variables())
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir=FLAGS.checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Session restored from trained model at {}!'.format(ckpt.model_checkpoint_path))
            else:
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), FLAGS.checkpoint_dir)

            total_v_utility = 0.0
            test_correct_num_utility = 0.0
            try:
                while not coord.should_stop():
                    videos, videos_labels = sess.run([videos_op, videos_labels_op])
                    feed = {videos_placeholder: videos, utility_labels_placeholder: videos_labels, dropout_placeholder: 1.0}
                    right_utility = sess.run(right_count_utility_op, feed_dict=feed)
                    print(total_v_utility)
                    test_correct_num_utility += right_utility
                    total_v_utility += videos_labels.shape[0]

            except tf.errors.OutOfRangeError:
                print('Done testing on all the examples')
            finally:
                coord.request_stop()

            print('Utility test acc:', test_correct_num_utility / total_v_utility,
                  'Utility test_correct_num:', test_correct_num_utility,
                  'Utility total_v:', total_v_utility)

            coord.join(threads)
            sess.close()

def linear_scaling(frame, vid, video_linear_scaling, joint_channels):
    if not video_linear_scaling:
        flst = []
        for i in range(frame.shape[2]):
            if joint_channels:
                max, min = np.max(frame), np.min(frame)
            else:
                max, min = np.max(frame[:, :, i]), np.min(frame[:, :, i])
            f = (frame[:, :, i] - min) / (max - min) * 255
            flst.append(f)
        frame = np.stack(flst, axis=2)
    else:
        flst = []
        for i in range(frame.shape[2]):
            if joint_channels:
                max, min = np.max(vid), np.min(vid)
            else:
                max, min = np.max(vid[:, :, :, i]), np.min(vid[:, :, :, i])
            f = (frame[:, :, i] - min) / (max - min) * 255
            flst.append(f)
        frame = np.stack(flst, axis=2)
    return frame

def plot_visualization(vid, vid_orig, heatmap_dir, plot_img=False, write_video=True):
    import cv2
    def crop_center(img, cropx, cropy):
        y, x,_ = img.shape
        startx = x // 2 - (cropx // 2)
        starty = y // 2 - (cropy // 2)
        return img[starty:starty + cropy, startx:startx + cropx, :]

    def plot_img_and_hist(image, axes, bins=256):
        #Plot an image along with its histogram and cumulative histogram.
        image = img_as_float(image)
        ax_img, ax_hist = axes

        ax_cdf = ax_hist.twinx()

        # Display image
        ax_img.imshow(image, cmap=plt.cm.gray)
        ax_img.set_axis_off()
        ax_img.set_adjustable('box-forced')

        # Display histogram
        ax_hist.hist(image.ravel(), bins=bins, histtype='step', color='black')
        ax_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
        ax_hist.set_xlabel('Pixel intensity')
        ax_hist.set_xlim(0, 1)
        ax_hist.set_yticks([])

        # Display cumulative distribution
        img_cdf, bins = exposure.cumulative_distribution(image, bins)
        ax_cdf.plot(bins, img_cdf, 'r')
        ax_cdf.set_yticks([])

        return ax_img, ax_hist, ax_cdf

    def PCA(data):
        m, n = data.shape[0], data.shape[1]
        #print(m, n)
        mean = np.mean(data, axis=0)
        data -= np.tile(mean, (m, 1))
        # calculate the covariance matrix
        cov = np.matmul(np.transpose(data), data)
        evals, evecs = np.linalg.eigh(cov)
        # sort eigenvalue in decreasing order
        idx = np.argsort(evals)[::-1]
        evecs = evecs[:, idx]
        evals = evals[idx]
        #print(evals)
        evecs = evecs[:, 0]
        return np.matmul(data, evecs), evals[0] / sum(evals)

    width, height = 1280, 720

    # if FLAGS.use_crop:
    #     height = FLAGS.crop_height
    #     width = FLAGS.crop_width
    # else:
    #     height = FLAGS.height
    #     width = FLAGS.width

    video_histeq = []
    for i in range(vid.shape[0]):
        print(i)
        #frame = crop_center(vid[i], 112, 112)
        frame = np.reshape(vid[i], (width * height, 3))
        frame, K = PCA(frame)
        frame = np.reshape(frame, (height, width))
        max, min = np.max(frame), np.min(frame)
        frame = ((frame - min) / (max - min) * 255).astype('uint8')
        # Contrast stretching
        p2, p98 = np.percentile(frame, (2, 98))
        img_rescale = exposure.rescale_intensity(frame, in_range=(p2, p98))
        # Equalization
        img_eq = exposure.equalize_hist(frame)
        video_histeq.append(img_eq)

        # Adaptive Equalization
        # img_adapteq = exposure.equalize_adapthist(frame, clip_limit=0.03)

        if plot_img:
            # Display results
            fig = plt.figure(figsize=(12, 16))
            axes = np.zeros((4, 3), dtype=np.object)
            axes[0, 0] = fig.add_subplot(4, 3, 1)
            for j in range(1, 3):
                axes[0, j] = fig.add_subplot(4, 3, 1 + j, sharex=axes[0, 0], sharey=axes[0, 0])
            for j in range(3, 12):
                axes[j // 3, j % 3] = fig.add_subplot(4, 3, 1 + j)

            ax_img, ax_hist, ax_cdf = plot_img_and_hist(frame, axes[0:2, 0])
            ax_img.set_title('PCA on 3 channels ({:.4f})'.format(K))

            y_min, y_max = ax_hist.get_ylim()
            ax_hist.set_ylabel('Number of pixels')
            ax_hist.set_yticks(np.linspace(0, y_max, 5))

            ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_rescale, axes[0:2, 1])
            ax_img.set_title('Contrast stretching')

            ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_eq, axes[0:2, 2])
            ax_img.set_title('Histogram equalization')

            #ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_adapteq, axes[0:1, 3])
            #ax_img.set_title('Adaptive equalization')

            ax_cdf.set_ylabel('Fraction of total intensity')
            ax_cdf.set_yticks(np.linspace(0, 1, 5))

            print(vid_orig[j].shape)
            frame_downsample_crop = crop_center(vid_orig[j], 112, 112)
            frame = crop_center(vid[j], 112, 112)
            axes[2, 0].imshow(frame_downsample_crop.astype('uint8'))
            axes[2, 0].set_title('Dowmsampled')
            frame_scaled_joint = linear_scaling(frame, vid, video_linear_scaling=True, joint_channels=True).astype('uint8')
            axes[2, 1].imshow(frame_scaled_joint.astype('uint8'))
            axes[2, 1].set_title('Joint Scaling')
            frame_scaled_separate = linear_scaling(frame, vid, video_linear_scaling=True, joint_channels=False).astype('uint8')
            axes[2, 2].imshow(frame_scaled_separate.astype('uint8'))
            axes[2, 2].set_title('Separate Scaling')
            for j in range(frame.shape[2]):
                axes[3, j].imshow(frame[:,:,j], cmap=plt.get_cmap('jet'))
                axes[3, j].set_title('Channel{}'.format(j))
            # prevent overlap of y-axis labels
            fig.tight_layout()
            plt.savefig('{}/vis_{}.png'.format(heatmap_dir, i))
            plt.close()
    print('writing video')
    if write_video:
        # fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Be sure to use lower case
        # output = "{}/hist_eq.avi".format(heatmap_dir)
        # out = cv2.VideoWriter(output, fourcc, 20.0, (width, height), False)
        # vid = np.multiply(np.asarray(video_histeq), 255).astype('uint8')
        # print(vid.shape)
        # print(output)
        # for i in range(vid.shape[0]):
        #     frame = vid[i]
        #     #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        #     #frame = frame.reshape(112, 112, 3)
        #     # print(frame)
        #     out.write(frame)
        # out.release()
        # cv2.destroyAllWindows()

        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Be sure to use lower case
        output = "{}/orig.avi".format(heatmap_dir)
        out = cv2.VideoWriter(output, fourcc, 30.0, (width, height), True)
        #vid_histeq = np.multiply(np.asarray(video_histeq), 255).astype('uint8')
        vid_histeq = vid_orig
        print(vid_histeq.shape)
        print(output)
        for i in range(vid_histeq.shape[0]):
            frame = vid_histeq[i]
            print(frame.shape)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            #frame = frame.reshape(112, 112, 3)
            print(frame)
            out.write(frame)
        out.release()
        #cv2.destroyAllWindows()

def read_ucf_labels(filename):
    label_dict = {}
    with open(filename, 'r') as f:
        for line in f:
            words = line.strip('\n').split()
            label_dict[int(words[0]) - 1] = words[1].lower()
    return label_dict

def visualize(directory, X, X_orig, Y):
    label_dict = read_ucf_labels('classInd.txt')
    smarthome_lst = ['wallpushups', 'boxingpunchingbag', 'handstandpushups', 'hulahoop', 'moppingfloor', 'pullups', 'pushups']
    for i in range(len(X)):
        print(label_dict[Y[i]])
        #if label_dict[Y[i]] in smarthome_lst:
        vis_dir = os.path.join(directory, '{}'.format(label_dict[Y[i]]))
        if not os.path.exists(vis_dir):
            os.makedirs(vis_dir)
        #else:
        #    continue
        vid = X[i]
        vid_orig = X_orig[i]
        plot_visualization(vid, vid_orig, vis_dir)
        #else:
        #    continue

def visualize_degradation_ucf():
    if not os.path.exists(FLAGS.visualization_dir):
        os.makedirs(FLAGS.visualization_dir)
    videos_placeholder, utility_labels_placeholder, budget_labels_placeholder, _, _ = placeholder_inputs(FLAGS.video_batch_size * FLAGS.gpu_num)
    videos_degraded_lst = []
    with tf.variable_scope(tf.get_variable_scope()) as scope:
        for gpu_index in range(0, FLAGS.gpu_num):
            with tf.device('/gpu:%d' % gpu_index):
                print('/gpu:%d' % gpu_index)
                with tf.name_scope('%s_%d' % ('gpu', gpu_index)) as scope:
                    videos = tf.reshape(
                        videos_placeholder[gpu_index * FLAGS.video_batch_size:(gpu_index + 1) * FLAGS.video_batch_size],
                        [FLAGS.video_batch_size * FLAGS.depth, FLAGS.crop_height, FLAGS.crop_width, FLAGS.nchannel])
                    with tf.variable_scope(tf.get_variable_scope(), reuse=False):
                        videos_degraded = residualNet(videos, training=False)
                    videos_degraded = tf.reshape(videos_degraded,
                                                [FLAGS.video_batch_size, FLAGS.depth, FLAGS.crop_height,
                                                 FLAGS.crop_width, FLAGS.nchannel])
                    videos_degraded_lst.append(videos_degraded)
                    # Reuse variables for the next tower.
                    tf.get_variable_scope().reuse_variables()
    videos_degraded_op = tf.concat(videos_degraded_lst, 0)
    train_files = [os.path.join(FLAGS.train_videos_files_dir, f) for
                   f in os.listdir(FLAGS.train_videos_files_dir) if f.endswith('.tfrecords')]
    val_files = [os.path.join(FLAGS.val_videos_files_dir, f) for
                 f in os.listdir(FLAGS.val_videos_files_dir) if f.endswith('.tfrecords')]

    print(train_files)
    print(val_files)
    videos_op, labels_op = inputs_videos(filenames=val_files,
                                         batch_size=FLAGS.gpu_num * FLAGS.video_batch_size,
                                         num_epochs=1,
                                         num_threads=FLAGS.num_threads,
                                         num_examples_per_epoch=FLAGS.num_examples_per_epoch,
                                         shuffle=False)

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        saver = tf.train.Saver(tf.trainable_variables())
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir=FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Session restored from trained model at {}!'.format(ckpt.model_checkpoint_path))
        else:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), FLAGS.checkpoint_dir)
        try:
            videos_degraded_lst = []
            labels_lst = []
            directory = FLAGS.visualization_dir
            while not coord.should_stop():
                videos, labels = sess.run([videos_op, labels_op])
                videos_degraded_value = sess.run(videos_degraded_op, feed_dict={videos_placeholder: videos})
                videos.tolist()
                videos_degraded_value.tolist()

                #videos_degraded_lst.append(videos_degraded_value*255)
                videos_degraded_lst.extend(videos_degraded_value)
                labels_lst.extend(labels)

                print('#####################################################')
                print(videos_degraded_value[0].shape)
                print(videos[0].shape)
                print(videos_degraded_value)
                print('#####################################################')

                visualize(directory, videos_degraded_value, videos, labels)
                #raise tf.errors.OutOfRangeError
        except tf.errors.OutOfRangeError:
            print('Done testing on all the examples')
        finally:
            coord.request_stop()
            coord.join(threads)

def visualize_degradation_ucf_smarthome():
    if not os.path.exists(FLAGS.visualization_dir):
        os.makedirs(FLAGS.visualization_dir)
    import cv2
    framearray = []
    #filename_lst = ['v_ShavingBeard_g18_c01.avi', 'v_ShavingBeard_g18_c02.avi', 'v_ShavingBeard_g18_c03.avi'
    #                'v_ShavingBeard_g18_c04.avi', 'v_ShavingBeard_g18_c05.avi', 'v_ShavingBeard_g18_c06.avi', 'v_ShavingBeard_g18_c07.avi']
    #filename_lst = ['Playing the Violin.mkv']

    #filename = "v_HandStandPushups_g04_c03.avi"
    #filename_lst = ["human_mask_horizental_only.avi"]
    filename_lst = ['TwoPushUp.mp4']
    for filename in filename_lst:
        cap = cv2.VideoCapture(filename)
        nframe = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        nframe = nframe - nframe % 2
        print(nframe)
        for i in range(nframe):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if frame is None:
                continue
            h, w, c = frame.shape
            print(h, w, c)
            #frame = cv2.resize(frame, (1280, 720))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            framearray.append(frame)
        cap.release()
    if len(framearray) % 2 != 0:
        framearray = framearray[:-1]
    # filename = "v_HandStandPushups_g04_c04.avi"
    # cap = cv2.VideoCapture(filename)
    # nframe = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # nframe = nframe - nframe % 16
    # for i in range(nframe):
    #     cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    #     ret, frame = cap.read()
    #     h, w, c = frame.shape
    #     print(h, w, c)
    #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #     framearray.append(frame)
    # cap.release()

    framearray = np.asarray(framearray)
    print(framearray.shape[0] // 2)
    videos = np.split(np.array(framearray), framearray.shape[0] // 2)
    print(videos[0].shape)
    print(len(videos))
    videos_placeholder = tf.placeholder(tf.float32, shape=(2, 720, 1280, 3))

    with tf.variable_scope(tf.get_variable_scope()) as scope:
        for gpu_index in range(0, 1):
            with tf.device('/gpu:%d' % gpu_index):
                print('/gpu:%d' % gpu_index)
                with tf.name_scope('%s_%d' % ('gpu', gpu_index)) as scope:
                    videos_degraded_op = residualNet(videos_placeholder, training=False)


    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        saver = tf.train.Saver(tf.trainable_variables())
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir=FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Session restored from trained model at {}!'.format(ckpt.model_checkpoint_path))
        else:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), FLAGS.checkpoint_dir)
        try:
            directory = FLAGS.visualization_dir
            videos_degraded_lst = []
            videos_lst = []
            for vid in videos:
                #print(vid)
                videos_degraded_value = sess.run(videos_degraded_op, feed_dict={videos_placeholder: vid})
                videos_degraded_lst.append(videos_degraded_value)
                videos_lst.append(vid)
            videos_degrad = np.asarray(videos_degraded_lst).reshape(len(videos_degraded_lst)*videos_degraded_value.shape[0], videos_degraded_value.shape[1], videos_degraded_value.shape[2], videos_degraded_value.shape[3])
            videos = np.asarray(videos_lst).reshape(len(videos_lst)*vid.shape[0], vid.shape[1], vid.shape[2], vid.shape[3])
            print(videos.shape)
            #print(videos_degrad.shape)
            visualize(directory, [videos_degrad], [videos], [37])
            #print(videos_degraded_value)
            #videos_degraded_lst.append(videos_degraded_value*255)
            #videos_degraded_lst.append(videos_degraded_value)
            #raise tf.errors.OutOfRangeError
        except tf.errors.OutOfRangeError:
            print('Done testing on all the examples')
        finally:
            coord.request_stop()
            coord.join(threads)

def visualize_degradation_video():
    videos_placeholder = tf.placeholder(tf.float32, shape=(
    FLAGS.video_batch_size * FLAGS.gpu_num, FLAGS.depth, None, None, FLAGS.nchannel))

    videos_degraded_lst = []
    with tf.variable_scope(tf.get_variable_scope()) as scope:
        for gpu_index in range(0, FLAGS.gpu_num):
            with tf.device('/gpu:%d' % gpu_index):
                print('/gpu:%d' % gpu_index)
                with tf.name_scope('%s_%d' % ('gpu', gpu_index)) as scope:
                    videos = tf.reshape(
                        videos_placeholder[gpu_index * FLAGS.video_batch_size:(gpu_index + 1) * FLAGS.video_batch_size],
                        [FLAGS.video_batch_size * FLAGS.depth, FLAGS.crop_height, FLAGS.crop_width, FLAGS.nchannel])
                    with tf.variable_scope(tf.get_variable_scope(), reuse=False):
                        videos_degraded = residualNet(videos, training=False)
                    videos_degraded = tf.reshape(videos_degraded,
                                                 [FLAGS.video_batch_size, FLAGS.depth, FLAGS.crop_height,
                                                  FLAGS.crop_width, FLAGS.nchannel])
                    videos_degraded_lst.append(videos_degraded)
                    # Reuse variables for the next tower.
                    tf.get_variable_scope().reuse_variables()
    videos_degraded_op = tf.concat(videos_degraded_lst, 0)
    train_files = [os.path.join(FLAGS.train_videos_files_dir, f) for
                   f in os.listdir(FLAGS.train_videos_files_dir) if f.endswith('.tfrecords')]
    val_files = [os.path.join(FLAGS.val_videos_files_dir, f) for
                 f in os.listdir(FLAGS.val_videos_files_dir) if f.endswith('.tfrecords')]

    print(train_files)
    print(val_files)
    videos_op, labels_op = inputs_videos(filenames=val_files,
                                         batch_size=FLAGS.gpu_num * FLAGS.video_batch_size,
                                         num_epochs=1,
                                         num_threads=FLAGS.num_threads,
                                         num_examples_per_epoch=FLAGS.num_examples_per_epoch,
                                         shuffle=False)

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        saver = tf.train.Saver(tf.trainable_variables())
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir=FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Session restored from trained model at {}!'.format(ckpt.model_checkpoint_path))
        else:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), FLAGS.checkpoint_dir)
        try:
            videos_degraded_lst = []
            labels_lst = []
            directory = FLAGS.visualization_dir
            while not coord.should_stop():
                videos, labels = sess.run([videos_op, labels_op])
                videos_degraded_value = sess.run(videos_degraded_op, feed_dict={videos_placeholder: videos})
                videos.tolist()
                videos_degraded_value.tolist()

                # videos_degraded_lst.append(videos_degraded_value*255)
                videos_degraded_lst.extend(videos_degraded_value)
                labels_lst.extend(labels)

                print('#####################################################')
                print(videos_degraded_value[0].shape)
                print(videos[0].shape)
                print(videos_degraded_value)
                print('#####################################################')

                visualize(directory, videos_degraded_value, videos, labels)
                # raise tf.errors.OutOfRangeError
        except tf.errors.OutOfRangeError:
            print('Done testing on all the examples')
        finally:
            coord.request_stop()
            coord.join(threads)

def visualize_degradation_vispr():
    images_placeholder = tf.placeholder(tf.float32,
                                        shape=(FLAGS.image_batch_size * FLAGS.gpu_num, None, None, FLAGS.nchannel))
    budget_labels_placeholder = tf.placeholder(tf.float32,
                                               shape=(FLAGS.image_batch_size * FLAGS.gpu_num, FLAGS.num_classes_budget))

    images_degraded_lst = []
    with tf.variable_scope(tf.get_variable_scope()) as scope:
        for gpu_index in range(0, FLAGS.gpu_num):
            with tf.device('/gpu:%d' % gpu_index):
                print('/gpu:%d' % gpu_index)
                with tf.name_scope('%s_%d' % ('gpu', gpu_index)) as scope:
                    X = tf.reshape(images_placeholder[gpu_index * FLAGS.image_batch_size:(gpu_index+1)*FLAGS.image_batch_size],
                                   [FLAGS.image_batch_size * 1, FLAGS.crop_height, FLAGS.crop_width, FLAGS.nchannel])
                    images_degraded = residualNet(X)
                    images_degraded = tf.reshape(images_degraded, [FLAGS.image_batch_size, 1, FLAGS.crop_height, FLAGS.crop_width,
                                              FLAGS.nchannel])
                    images_degraded_lst.append(images_degraded)
                    # Reuse variables for the next tower.
                    tf.get_variable_scope().reuse_variables()
    images_degraded_op = tf.concat(images_degraded_lst, 0)

    train_image_files = [os.path.join(FLAGS.train_images_files_dir, f) for f in
                         os.listdir(FLAGS.train_images_files_dir) if f.endswith('.tfrecords')]
    print(train_image_files)
    val_image_files = [os.path.join(FLAGS.val_images_files_dir, f) for f in
                       os.listdir(FLAGS.val_images_files_dir) if f.endswith('.tfrecords')]
    print(val_image_files)
    images_op, images_labels_op = inputs_images(filenames=val_image_files,
                                                                   batch_size=FLAGS.image_batch_size * FLAGS.gpu_num,
                                                                   num_epochs=None,
                                                                   num_threads=FLAGS.num_threads,
                                                                   num_examples_per_epoch=FLAGS.num_examples_per_epoch,
                                                                   shuffle=True)

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        saver = tf.train.Saver(tf.trainable_variables())
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir=FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Session restored from trained model at {}!'.format(ckpt.model_checkpoint_path))
        else:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), FLAGS.checkpoint_dir)
        try:
            images_degraded_lst = []
            labels_lst = []
            directory = 'visualization'
            while not coord.should_stop():
                images, images_labels = sess.run(
                    [images_op, images_labels_op])
                print(images)
                images_degraded_value = sess.run(images_degraded_op, feed_dict={images_placeholder: images})
                images.tolist()
                images_degraded_value.tolist()

                #videos_degraded_lst.append(videos_degraded_value*255)
                images_degraded_lst.extend(images_degraded_value)
                labels_lst.extend(images_labels)

                print('#####################################################')
                print(images_degraded_value[0].shape)
                print(images[0].shape)
                #print(images_degraded_value)
                print('#####################################################')

                visualize(directory, images_degraded_value, images, images_labels)
                #raise tf.errors.OutOfRangeError
        except tf.errors.OutOfRangeError:
            print('Done testing on all the examples')
        finally:
            coord.request_stop()
            coord.join(threads)

def convert_to(images, labels, name, directory):
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _floats_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

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

def merge_tfrecords():
    images_placeholder = tf.placeholder(tf.float32, shape=(FLAGS.image_batch_size * FLAGS.gpu_num, None, None, FLAGS.nchannel))
    images_degraded_lst = []
    with tf.variable_scope(tf.get_variable_scope()) as scope:
        for gpu_index in range(0, FLAGS.gpu_num):
            with tf.device('/gpu:%d' % gpu_index):
                print('/gpu:%d' % gpu_index)
                with tf.name_scope('%s_%d' % ('gpu', gpu_index)) as scope:
                    LR_images = residualNet(
                        images_placeholder[gpu_index * FLAGS.image_batch_size:(gpu_index + 1) * FLAGS.image_batch_size])

                    images_degraded_lst.append(LR_images)
                    # Reuse variables for the next tower.
                    tf.get_variable_scope().reuse_variables()
    images_degraded_op = tf.concat(images_degraded_lst, 0)

    train_image_files = [os.path.join(FLAGS.train_images_files_dir, f) for f in
                         os.listdir(FLAGS.train_images_files_dir) if f.endswith('.tfrecords')]
    print(train_image_files)
    val_image_files = [os.path.join(FLAGS.val_images_files_dir, f) for f in
                       os.listdir(FLAGS.val_images_files_dir) if f.endswith('.tfrecords')]
    print(val_image_files)
    test_image_files = [os.path.join(FLAGS.test_images_files_dir, f) for f in
                        os.listdir(FLAGS.test_images_files_dir) if f.endswith('.tfrecords')]
    print(test_image_files)

    images_op, images_labels_op = inputs_images(filenames=test_image_files,
                                                           batch_size=FLAGS.image_batch_size * FLAGS.gpu_num,
                                                           num_epochs=1,
                                                           num_threads=FLAGS.num_threads,
                                                           num_examples_per_epoch=FLAGS.num_examples_per_epoch,
                                                           shuffle=False)

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        saver = tf.train.Saver(tf.trainable_variables())
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir=FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Session restored from trained model at {}!'.format(ckpt.model_checkpoint_path))
        else:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), FLAGS.checkpoint_dir)
        try:
            k = 0
            images_degraded_lst = []
            images_labels_lst = []
            while not coord.should_stop():
                images, images_labels = sess.run([images_op, images_labels_op])
                images_degraded_value = sess.run(images_degraded_op, feed_dict={images_placeholder: images})
                #print(images_degraded_value.shape)
                #print(images_labels.shape)
                images_degraded_lst.append(images_degraded_value)
                images_labels_lst.append(images_labels)
                k += 1
                print(k)
        except tf.errors.OutOfRangeError:
            print('Done testing on all the examples')
        finally:
            coord.request_stop()
        coord.join(threads)

        directory = 'tfrecords'
        X, Y = np.asarray(np.concatenate(images_degraded_lst, axis=0)).astype(np.float32), np.asarray(np.concatenate(images_labels_lst,axis=0)).astype(np.float32)
        print(X.shape)
        print(Y.shape)
        convert_to(X, Y, 'testing_{}'.format(X.shape[0]), directory)

def main(_):
    pp = pprint.PrettyPrinter()
    pp.pprint(FLAGS.__flags)
    visualize_degradation_ucf_smarthome()
    #run_testing_vispr()
    #run_testing_ucf101()
    #run_training_multi_model_residual()
    #run_testing_multi_model_residual_vispr()
    #run_testing_multi_model_residual_ucf()
    #run_training_multi_model()
    #run_testing_multi_model()
    #run_training()
    #run_testing_vispr()
    #run_testing_ucf101()
    #run_testing()
    #run_training_degradation()
    #run_pretraining()
    #run_testing_degradation()
    #run_training_utility()
    #run_testing_utility()
    #merge_tfrecords()
    #run_pretraining()
    #visualize_degradation_ucf()
    #visualize_degradation()
    #visualize_degradation_vispr()

if __name__ == '__main__':
    tf.app.run()
