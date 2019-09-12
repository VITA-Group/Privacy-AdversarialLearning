# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Generic training script that trains a model using a given dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from nets import nets_factory
import time
import os
import errno
from six.moves import xrange
import numpy as np
import cv2
import sys
import utils
#from torchvision import transforms
from PIL import Image
#import datasets
#import torch
from hopenet import hopenet
slim = tf.contrib.slim
import input_data
from tf_flags import FLAGS
from residualnet import residualnet
from mobilenet import mobilenet
from utils import ckpt_path_map, model_name_mapping
import itertools

os.environ["CUDA_VISIBLE_DEVICES"]="2"

def accuracy(logits, labels):
    correct_pred = tf.equal(tf.argmax(tf.nn.softmax(logits), axis=1), labels)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return accuracy

def tower_loss_mse(preds, labels):
    labels = tf.cast(labels, tf.float32)
    MSE = tf.reduce_mean(
            tf.square(labels - preds))
    return MSE

def tower_loss_xentropy_dense(logits, labels):
    labels = tf.cast(labels, tf.float32)
    xentropy_mean = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    return xentropy_mean

def tower_loss_xentropy_sparse(logits, labels):
    labels = tf.cast(labels, tf.int64)
    xentropy_mean = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    return xentropy_mean

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

def placeholder_inputs(batch_size):
    images_placeholder = tf.placeholder(tf.float32, shape=(batch_size, 224, 224, 3))
    label_yaw_placeholder = tf.placeholder(tf.float32, shape=(batch_size))
    label_pitch_placeholder = tf.placeholder(tf.float32, shape=(batch_size))
    label_roll_placeholder = tf.placeholder(tf.float32, shape=(batch_size))
    label_yaw_cont_placeholder = tf.placeholder(tf.float32, shape=(batch_size))
    label_pitch_cont_placeholder = tf.placeholder(tf.float32, shape=(batch_size))
    label_roll_cont_placeholder = tf.placeholder(tf.float32, shape=(batch_size))
    return images_placeholder, label_yaw_placeholder, label_pitch_placeholder, label_roll_placeholder, \
           label_yaw_cont_placeholder, label_pitch_cont_placeholder, label_roll_cont_placeholder

def run_training_degradation_residual():
    if not os.path.exists(FLAGS.degradation_models):
        os.makedirs(FLAGS.degradation_models)
    start_from_trained_model = False

    with tf.Graph().as_default():
        global_step = tf.get_variable(
                    'global_step',
                    [],
                    initializer=tf.constant_initializer(0),
                    trainable=False
                    )
        images_placeholder = tf.placeholder(tf.float32, [FLAGS.batch_size*FLAGS.gpu_num, 224, 224, 3], name='images')
        labels_placeholder = tf.placeholder(tf.float32, [FLAGS.batch_size*FLAGS.gpu_num, 224, 224, 3], name='labels')
        tower_grads = []
        losses = []
        opt = tf.train.AdamOptimizer(1e-4)

        with tf.variable_scope(tf.get_variable_scope()) as scope:
            for gpu_index in range(0, FLAGS.gpu_num):
                with tf.device('/gpu:%d' % gpu_index):
                    print('/gpu:%d' % gpu_index)
                    with tf.name_scope('%s_%d' % ('gpu', gpu_index)) as scope:
                        pred = residualnet(images_placeholder[gpu_index * FLAGS.batch_size:(gpu_index + 1) * FLAGS.batch_size])
                        loss = tower_loss_mse(pred,
                                          labels_placeholder[gpu_index * FLAGS.batch_size:(gpu_index + 1) * FLAGS.batch_size])
                        losses.append(loss)
                        grads = opt.compute_gradients(loss)
                        tower_grads.append(grads)
                        tf.get_variable_scope().reuse_variables()
        loss_op = tf.reduce_mean(losses, name='mse')
        psnr_op = tf.multiply(tf.constant(20, dtype=tf.float32), tf.log(1 /tf.sqrt(loss_op))/tf.log(tf.constant(10, dtype=tf.float32)), name='psnr')

        grads_avg = average_gradients(tower_grads)
        apply_gradient_op = opt.apply_gradients(grads_avg, global_step=global_step)
        # Stochastic gradient descent with the standard backpropagation
        train_op = apply_gradient_op

        train_files = [os.path.join(FLAGS.train_files_dir, f) for f in
                       os.listdir(FLAGS.train_files_dir) if f.endswith('.tfrecords')]
        val_files = [os.path.join(FLAGS.val_files_dir, f) for f in
                     os.listdir(FLAGS.val_files_dir) if f.endswith('.tfrecords')]
        tr_images_op, _, _, _, _, _, _, _ = input_data.inputs_images(filenames = train_files,
                                                 batch_size=FLAGS.batch_size * FLAGS.gpu_num,
                                                 num_epochs=None,
                                                 num_threads=FLAGS.num_threads,
                                                 num_examples_per_epoch=FLAGS.num_examples_per_epoch)
        val_images_op, _, _, _, _, _, _, _ = input_data.inputs_images(filenames = val_files,
                                                   batch_size=FLAGS.batch_size * FLAGS.gpu_num,
                                                   num_epochs=None,
                                                   num_threads=FLAGS.num_threads,
                                                   num_examples_per_epoch=FLAGS.num_examples_per_epoch)

        conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        conf.gpu_options.allow_growth = True
        sess = tf.Session(config=conf)
        init_op = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        gvar_list = tf.global_variables()
        bn_moving_vars = [g for g in gvar_list if 'moving_mean' in g.name]
        bn_moving_vars += [g for g in gvar_list if 'moving_variance' in g.name]
        print(bn_moving_vars)
        if start_from_trained_model:
            #vardict = {v.name[18:-2]: v for v in tf.trainable_variables()}
            saver = tf.train.Saver(tf.trainable_variables() + bn_moving_vars)
            #print(vardict)
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir=FLAGS.degradation_models)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Session Restored!')
            else:
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), FLAGS.degradation_models)

        print("Training...")
        saver = tf.train.Saver(tf.trainable_variables())
        for step in range(FLAGS.max_steps):
            # Run by batch images
            start_time = time.time()
            tr_images = sess.run(tr_images_op)
            tr_labels = np.empty_like(tr_images)
            tr_labels[:] = tr_images
            print(tr_images.shape)
            print(tr_labels.shape)
            tr_feed = {images_placeholder: tr_images, labels_placeholder: tr_labels}
            _, loss_value = sess.run([train_op, loss_op], feed_dict=tr_feed)
            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
            print("Step: [%2d], time: [%4.4f], training_loss = [%.8f]" % (step, time.time()-start_time, loss_value))
            if step % FLAGS.val_step == 0:
                val_images = sess.run(val_images_op)
                val_labels = np.empty_like(val_images)
                val_labels[:] = val_images
                val_feed = {images_placeholder: val_images, labels_placeholder: val_labels}
                loss_value, psnr = sess.run([loss_op, psnr_op], feed_dict=val_feed)
                print("Step: [%2d], time: [%4.4f], validation_loss = [%.8f], validation_psnr = [%.8f]" %
                      (step, time.time()-start_time, loss_value, psnr))
                tr_images = sess.run(tr_images_op)
                tr_labels = np.empty_like(tr_images)
                tr_labels[:] = tr_images
                tr_feed = {images_placeholder: tr_images, labels_placeholder: tr_labels}
                loss_value, psnr = sess.run([loss_op, psnr_op], feed_dict=tr_feed)
                print("Step: [%2d], time: [%4.4f], training_loss = [%.8f], training_psnr = [%.8f]" %
                      (step, time.time()-start_time, loss_value, psnr))
            if step % FLAGS.save_step == 0 or (step+1) == FLAGS.max_steps:
                checkpoint_path = os.path.join(FLAGS.degradation_models, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

        coord.request_stop()
        coord.join(threads)
        sess.close()

def run_pretraining():
    # Create model directory
    if not os.path.exists(FLAGS.whole_pretraining):
        os.makedirs(FLAGS.whole_pretraining)

    use_pretrained_model = False
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True

    #multiplier_lst = [0.5 - i * 0.02 for i in range(FLAGS.NBudget)]

    with tf.Graph().as_default():
        with tf.Session(config=config) as sess:
            global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
            images_placeholder, label_yaw_placeholder, label_pitch_placeholder, label_roll_placeholder, label_yaw_cont_placeholder, label_pitch_cont_placeholder, label_roll_cont_placeholder = placeholder_inputs(
                FLAGS.batch_size * FLAGS.gpu_num)
            istraining_placeholder = tf.placeholder(tf.bool)


            losses_utility = []
            losses_yaw = []
            losses_pitch = []
            losses_roll = []
            tower_grads_degradation = []
            tower_grads_utility = []


            opt_degradation = tf.train.AdamOptimizer(1e-5)
            opt_utility = tf.train.AdamOptimizer(1e-5)

            with tf.variable_scope(tf.get_variable_scope()) as scope:
                for gpu_index in range(0, FLAGS.gpu_num):
                    with tf.device('/gpu:%d' % gpu_index):
                        print('/gpu:%d' % gpu_index)
                        with tf.name_scope('%s_%d' % ('gpu', gpu_index)) as scope:
                            X = images_placeholder[gpu_index * FLAGS.batch_size:(gpu_index + 1) * FLAGS.batch_size]
                            X = residualnet(X)

                            fc_yaw, fc_pitch, fc_roll = hopenet(X, is_training=istraining_placeholder)

                            loss_yaw = tower_loss_xentropy_sparse(fc_yaw,
                                            label_yaw_placeholder[gpu_index * FLAGS.batch_size:(gpu_index + 1) * FLAGS.batch_size])
                            loss_pitch = tower_loss_xentropy_sparse(fc_pitch,
                                            label_pitch_placeholder[gpu_index * FLAGS.batch_size:(gpu_index + 1) * FLAGS.batch_size])
                            loss_roll = tower_loss_xentropy_sparse(fc_roll,
                                            label_roll_placeholder[gpu_index * FLAGS.batch_size:(gpu_index + 1) * FLAGS.batch_size])

                            idx_tensor = tf.expand_dims(tf.range(66, dtype=tf.float32), 0)
                            idx_tensor = tf.tile(idx_tensor, [FLAGS.batch_size, 1])

                            yaw_predicted = tf.reduce_sum(tf.multiply(tf.nn.softmax(fc_yaw, axis=1), idx_tensor),
                                                          axis=1) * 3 - 99
                            pitch_predicted = tf.reduce_sum(tf.multiply(tf.nn.softmax(fc_pitch, axis=1), idx_tensor),
                                                            axis=1) * 3 - 99
                            roll_predicted = tf.reduce_sum(tf.multiply(tf.nn.softmax(fc_roll, axis=1), idx_tensor),
                                                           axis=1) * 3 - 99

                            loss_reg_yaw = tower_loss_mse(yaw_predicted,
                                            label_yaw_cont_placeholder[gpu_index * FLAGS.batch_size:(gpu_index + 1) * FLAGS.batch_size])
                            loss_reg_pitch = tower_loss_mse(pitch_predicted,
                                            label_pitch_cont_placeholder[gpu_index * FLAGS.batch_size:(gpu_index + 1) * FLAGS.batch_size])
                            loss_reg_roll = tower_loss_mse(roll_predicted,
                                            label_roll_cont_placeholder[gpu_index * FLAGS.batch_size:(gpu_index + 1) * FLAGS.batch_size])

                            loss_yaw += FLAGS.alpha * loss_reg_yaw
                            loss_pitch += FLAGS.alpha * loss_reg_pitch
                            loss_roll += FLAGS.alpha * loss_reg_roll

                            loss_utility = loss_yaw + loss_pitch + loss_roll
                            losses_utility.append(loss_utility)
                            losses_yaw.append(loss_yaw)
                            losses_pitch.append(loss_pitch)
                            losses_roll.append(loss_roll)

                            varlist_degradtion = [v for v in tf.trainable_variables() if any(x in v.name for x in ["DegradationModule"])]
                            print([v.name for v in varlist_degradtion])
                            varlist_utility = [v for v in tf.trainable_variables() if any(x in v.name for x in ["UtilityModule"])]
                            print([v.name for v in varlist_utility])

                            grads_degradation = opt_degradation.compute_gradients(loss_utility, varlist_degradtion)
                            grads_utility = opt_utility.compute_gradients(loss_utility, varlist_utility)

                            tower_grads_degradation.append(grads_degradation)
                            #tower_grads_budget.append(grads_budget)
                            tower_grads_utility.append(grads_utility)

                            # Reuse variables for the next tower.
                            tf.get_variable_scope().reuse_variables()

            loss_utility_op = tf.reduce_mean(losses_utility, name='loss_utility')
            #loss_budget_op = tf.reduce_mean(loss_budget_lst, name='softmax')
            loss_yaw_op = tf.reduce_mean(losses_yaw, name='loss_yaw')
            loss_pitch_op = tf.reduce_mean(losses_pitch, name='loss_pitch')
            loss_roll_op = tf.reduce_mean(losses_roll, name='loss_roll')

            grads_degradation = average_gradients(tower_grads_degradation)
            grads_utility = average_gradients(tower_grads_utility)

            with tf.device('/cpu:%d' % 0):
                tvs_degradation = varlist_degradtion
                accum_vars_degradtion =  [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in tvs_degradation]
                zero_ops_degradation = [tv.assign(tf.zeros_like(tv)) for tv in accum_vars_degradtion]

            # with tf.device('/cpu:%d' % 0):
            #     tvs_budget = varlist_budget
            #     accum_vars_budget =  [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in tvs_budget]
            #     zero_ops_budget = [tv.assign(tf.zeros_like(tv)) for tv in accum_vars_budget]

            with tf.device('/cpu:%d' % 0):
                tvs_utility = varlist_utility
                accum_vars_utility =  [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in tvs_utility]
                zero_ops_utility = [tv.assign(tf.zeros_like(tv)) for tv in accum_vars_utility]


            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies([tf.group(*update_ops)]):
                accum_ops_degradation = [accum_vars_degradtion[i].assign_add(gv[0] / FLAGS.n_minibatches) for i, gv in
                                         enumerate(grads_degradation)]
                accum_ops_utility = [accum_vars_utility[i].assign_add(gv[0] / FLAGS.n_minibatches) for i, gv
                                          in enumerate(grads_utility)]

            apply_gradient_op_degradation = opt_degradation.apply_gradients(
                    [(accum_vars_degradtion[i].value(), gv[1]) for i, gv in enumerate(grads_degradation)],
                    global_step=global_step)
            apply_gradient_op_utility = opt_utility.apply_gradients(
                    [(accum_vars_utility[i].value(), gv[1]) for i, gv in enumerate(grads_utility)],
                    global_step=global_step)


            train_op = tf.group(apply_gradient_op_degradation, apply_gradient_op_utility)

            train_files = [os.path.join(FLAGS.train_files_dir, f) for f in
                               os.listdir(FLAGS.train_files_dir) if f.endswith('.tfrecords')]
            val_files = [os.path.join(FLAGS.val_files_dir, f) for f in
                             os.listdir(FLAGS.val_files_dir) if f.endswith('.tfrecords')]


            print(train_files)
            print(val_files)

            tr_images_op, tr_labels_yaw_op, tr_labels_pitch_op, tr_labels_roll_op, \
            tr_labels_yaw_cont_op, tr_labels_pitch_cont_op, tr_labels_roll_cont_op, _ = input_data.inputs_images(filenames=train_files,
                                                                  batch_size=FLAGS.batch_size * FLAGS.gpu_num,
                                                                  num_epochs=None,
                                                                  num_threads=FLAGS.num_threads,
                                                                  num_examples_per_epoch=FLAGS.num_examples_per_epoch)

            val_images_op, val_labels_yaw_op, val_labels_pitch_op, val_labels_roll_op, \
            val_labels_yaw_cont_op, val_labels_pitch_cont_op, val_labels_roll_cont_op, _ = input_data.inputs_images(filenames=val_files,
                                                                  batch_size=FLAGS.batch_size * FLAGS.gpu_num,
                                                                  num_epochs=None,
                                                                  num_threads=FLAGS.num_threads,
                                                                  num_examples_per_epoch=FLAGS.num_examples_per_epoch)

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
                    print('Session restored from pretrained model at {}!'.format(ckpt.model_checkpoint_path))
                else:
                    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), ckpt_dir)

            gvar_list = tf.global_variables()
            bn_moving_vars = [g for g in gvar_list if 'moving_mean' in g.name]
            bn_moving_vars += [g for g in gvar_list if 'moving_variance' in g.name]
            # Create a saver for writing training checkpoints.
            if use_pretrained_model:
                varlist = [v for v in tf.trainable_variables() if any(x in v.name.split('/')[0] for x in ["DegradationModule"])]
                restore_model_ckpt(FLAGS.degradation_models, varlist, "DegradationModule")

                varlist = [v for v in tf.trainable_variables() + bn_moving_vars if any(x in v.name.split('/')[0] for x in ["UtilityModule"])]
                restore_model_ckpt(FLAGS.utility_models, varlist, "UtilityModule")
            else:
                varlist = tf.trainable_variables() + bn_moving_vars
                saver = tf.train.Saver(varlist)
                ckpt = tf.train.get_checkpoint_state(checkpoint_dir=FLAGS.whole_pretraining)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    print('Session restored from trained model at {}!'.format(ckpt.model_checkpoint_path))
                else:
                    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), FLAGS.whole_pretraining)

            saver = tf.train.Saver(tf.trainable_variables() + bn_moving_vars)

            for step in xrange(FLAGS.pretraining_steps):
                start_time = time.time()
                sess.run([zero_ops_utility, zero_ops_degradation])
                loss_utility_lst = []
                loss_yaw_lst = []
                loss_pitch_lst = []
                loss_roll_lst = []
                for _ in itertools.repeat(None, FLAGS.n_minibatches):
                    images, labels_yaw, labels_pitch, labels_roll, labels_yaw_cont, \
                    labels_pitch_cont, labels_roll_cont = sess.run([tr_images_op,
                                                                    tr_labels_yaw_op, tr_labels_pitch_op,
                                                                    tr_labels_roll_op,
                                                                    tr_labels_yaw_cont_op, tr_labels_pitch_cont_op,
                                                                    tr_labels_roll_cont_op])
                    _, _, loss_utility, loss_yaw, loss_pitch, loss_roll = sess.run([accum_ops_utility, accum_ops_degradation, loss_utility_op, loss_yaw_op, loss_pitch_op, loss_roll_op],
                                                                  feed_dict={images_placeholder: images,
                                                                             label_yaw_placeholder: labels_yaw,
                                                                             label_pitch_placeholder: labels_pitch,
                                                                             label_roll_placeholder: labels_roll,
                                                                             label_yaw_cont_placeholder: labels_yaw_cont,
                                                                             label_pitch_cont_placeholder: labels_pitch_cont,
                                                                             label_roll_cont_placeholder: labels_roll_cont,
                                                                             istraining_placeholder: True})
                    loss_utility_lst.append(loss_utility)
                    loss_yaw_lst.append(loss_yaw)
                    loss_pitch_lst.append(loss_pitch)
                    loss_roll_lst.append(loss_roll)
                sess.run(train_op)
                loss_summary = '(Training) Utility Module + Degradation Module, Step: {:4d}, time: {:.4f}, Utility loss: {:.8f}, Yaw {:.4f}, Pitch {:.4f}, Roll {:.4f}'.format(
                            step,
                            time.time() - start_time,
                            np.mean(loss_utility_lst),
                            np.mean(loss_yaw_lst),
                            np.mean(loss_pitch_lst),
                            np.mean(loss_roll_lst)
                )
                print(loss_summary)

                if step % FLAGS.val_step == 0:
                    start_time = time.time()
                    loss_utility_lst = []
                    loss_yaw_lst = []
                    loss_pitch_lst = []
                    loss_roll_lst = []
                    for _ in itertools.repeat(None, FLAGS.n_minibatches):
                        images, labels_yaw, labels_pitch, labels_roll, labels_yaw_cont, \
                        labels_pitch_cont, labels_roll_cont = sess.run([tr_images_op,
                                                                        tr_labels_yaw_op, tr_labels_pitch_op,
                                                                        tr_labels_roll_op,
                                                                        tr_labels_yaw_cont_op, tr_labels_pitch_cont_op,
                                                                        tr_labels_roll_cont_op])
                        loss_utility, loss_yaw, loss_pitch, loss_roll = sess.run([loss_utility_op, loss_yaw_op, loss_pitch_op, loss_roll_op],
                                                            feed_dict={images_placeholder: images,
                                                                       label_yaw_placeholder: labels_yaw,
                                                                       label_pitch_placeholder: labels_pitch,
                                                                       label_roll_placeholder: labels_roll,
                                                                       label_yaw_cont_placeholder: labels_yaw_cont,
                                                                       label_pitch_cont_placeholder: labels_pitch_cont,
                                                                       label_roll_cont_placeholder: labels_roll_cont,
                                                                       istraining_placeholder: False})
                        loss_utility_lst.append(loss_utility)
                        loss_yaw_lst.append(loss_yaw)
                        loss_pitch_lst.append(loss_pitch)
                        loss_roll_lst.append(loss_roll)
                    loss_summary = '(Training Evaluation) Step: {:4d}, time: {:.4f}, Utility loss: {:.8f}, Yaw {:.4f}, Pitch {:.4f}, Roll {:.4f}'.format(
                        step,
                        time.time() - start_time,
                        np.mean(loss_utility_lst),
                        np.mean(loss_yaw_lst),
                        np.mean(loss_pitch_lst),
                        np.mean(loss_roll_lst)
                    )
                    print(loss_summary)

                    start_time = time.time()
                    loss_utility_lst = []
                    loss_yaw_lst = []
                    loss_pitch_lst = []
                    loss_roll_lst = []
                    for _ in itertools.repeat(None, FLAGS.n_minibatches):
                        images, labels_yaw, labels_pitch, labels_roll, labels_yaw_cont, \
                        labels_pitch_cont, labels_roll_cont = sess.run([val_images_op,
                                                                        val_labels_yaw_op, val_labels_pitch_op,
                                                                        val_labels_roll_op,
                                                                        val_labels_yaw_cont_op, val_labels_pitch_cont_op,
                                                                        val_labels_roll_cont_op])
                        loss_utility, loss_yaw, loss_pitch, loss_roll = sess.run(
                            [loss_utility_op, loss_yaw_op, loss_pitch_op, loss_roll_op],
                            feed_dict={images_placeholder: images,
                                       label_yaw_placeholder: labels_yaw,
                                       label_pitch_placeholder: labels_pitch,
                                       label_roll_placeholder: labels_roll,
                                       label_yaw_cont_placeholder: labels_yaw_cont,
                                       label_pitch_cont_placeholder: labels_pitch_cont,
                                       label_roll_cont_placeholder: labels_roll_cont,
                                       istraining_placeholder: False})
                        loss_utility_lst.append(loss_utility)
                        loss_yaw_lst.append(loss_yaw)
                        loss_pitch_lst.append(loss_pitch)
                        loss_roll_lst.append(loss_roll)
                    loss_summary = '(Validation Evaluation) Step: {:4d}, time: {:.4f}, Utility loss: {:.8f}, Yaw {:.4f}, Pitch {:.4f}, Roll {:.4f}'.format(
                        step,
                        time.time() - start_time,
                        np.mean(loss_utility_lst),
                        np.mean(loss_yaw_lst),
                        np.mean(loss_pitch_lst),
                        np.mean(loss_roll_lst)
                    )
                    print(loss_summary)

                if step % FLAGS.save_step == 0 or (step + 1) == FLAGS.max_steps:
                    checkpoint_path = os.path.join(FLAGS.whole_pretraining, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)

            coord.request_stop()
            coord.join(threads)
        print("done")

def _avg_replicate(X):
    X = tf.reduce_mean(X, axis=3, keep_dims=True)
    X = tf.tile(X, [1, 1, 1, 3])
    return X

def run_training_multi_model_mobilenet_restarting():
    # Create model directory
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    #if not os.path.exists(FLAGS.saved_checkpoint_dir):
    #    os.makedirs(FLAGS.saved_checkpoint_dir)

    use_whole_pretrained_model = True
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True

    multiplier_lst = [0.5 - i * 0.02 for i in range(FLAGS.NBudget)]
    from collections import defaultdict
    logits_budget_lst_dct = defaultdict(list)
    loss_budget_lst_dct = defaultdict(list)
    wplaceholder_dct = {}

    with tf.Graph().as_default():
        with tf.Session(config=config) as sess:
            global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
            images_placeholder, label_yaw_placeholder, label_pitch_placeholder, label_roll_placeholder, label_yaw_cont_placeholder, label_pitch_cont_placeholder, label_roll_cont_placeholder = placeholder_inputs(
                FLAGS.batch_size * FLAGS.gpu_num)
            budget_labels_placeholder =  tf.placeholder(tf.int64, shape=(FLAGS.batch_size * FLAGS.gpu_num))

            budget_uniform_labels_placeholder = tf.placeholder(tf.float32, shape=(FLAGS.batch_size * FLAGS.gpu_num, FLAGS.num_classes_budget))
            istraining_placeholder = tf.placeholder(tf.bool)
            for multiplier in multiplier_lst:
                wplaceholder_dct['{}'.format(multiplier)] = tf.placeholder(tf.float32)

            L2_losses = []
            losses_utility = []
            losses_yaw = []
            losses_pitch = []
            losses_roll = []

            tower_grads_degradation = []
            tower_grads_utility = []
            tower_grads_budget = []

            # Compute Acc
            logits_budget_lst = []

            # Compute Loss
            losses_budget = []
            losses_degrad = []

            # Compute prediction with min entropy (most confident)
            # Use max uniform loss instead
            min_centpy_lst = []
            argmax_centpy_lst = []
            loss_budget_uniform_tensor_lst = []

            opt_degradation = tf.train.AdamOptimizer(FLAGS.degradation_lr)
            opt_utility = tf.train.AdamOptimizer(FLAGS.utility_lr)
            opt_budget = tf.train.AdamOptimizer(FLAGS.budget_lr)

            if FLAGS.use_lambda_decay:
                _lambda_op = tf.train.exponential_decay(FLAGS._lambda, global_step=global_step, decay_steps=10, decay_rate=0.9)
            else:
                _lambda_op = tf.identity(FLAGS._lambda)
            with tf.variable_scope(tf.get_variable_scope()) as scope:
                for gpu_index in range(0, FLAGS.gpu_num):
                    with tf.device('/gpu:%d' % gpu_index):
                        print('/gpu:%d' % gpu_index)
                        with tf.name_scope('%s_%d' % ('gpu', gpu_index)) as scope:
                            images_degraded = residualnet(images_placeholder[gpu_index * FLAGS.batch_size:(gpu_index + 1) * FLAGS.batch_size])
                            L2_loss = tower_loss_mse(images_degraded, images_placeholder[gpu_index * FLAGS.batch_size:(gpu_index + 1) * FLAGS.batch_size])
                            L2_losses.append(L2_loss)

                            L1_loss = tf.reduce_mean(tf.abs(images_degraded - images_placeholder[gpu_index * FLAGS.batch_size:(gpu_index + 1) * FLAGS.batch_size]))

                            if FLAGS.use_avg_replicate:
                                images_degraded = _avg_replicate(images_degraded)

                            fc_yaw, fc_pitch, fc_roll = hopenet(images_degraded, is_training=istraining_placeholder)
                            loss_yaw = tower_loss_xentropy_sparse(fc_yaw,
                                                                  label_yaw_placeholder[gpu_index * FLAGS.batch_size:(gpu_index + 1) * FLAGS.batch_size])
                            loss_pitch = tower_loss_xentropy_sparse(fc_pitch,
                                                                  label_pitch_placeholder[gpu_index * FLAGS.batch_size:(gpu_index + 1) * FLAGS.batch_size])
                            loss_roll = tower_loss_xentropy_sparse(fc_roll,
                                                                   label_roll_placeholder[gpu_index * FLAGS.batch_size:(gpu_index + 1) * FLAGS.batch_size])

                            idx_tensor = tf.expand_dims(tf.range(66, dtype=tf.float32), 0)
                            idx_tensor = tf.tile(idx_tensor, [FLAGS.batch_size, 1])

                            yaw_predicted = tf.reduce_sum(tf.multiply(tf.nn.softmax(fc_yaw, axis=1), idx_tensor),
                                                          axis=1) * 3 - 99
                            pitch_predicted = tf.reduce_sum(tf.multiply(tf.nn.softmax(fc_pitch, axis=1), idx_tensor),
                                                            axis=1) * 3 - 99
                            roll_predicted = tf.reduce_sum(tf.multiply(tf.nn.softmax(fc_roll, axis=1), idx_tensor),
                                                           axis=1) * 3 - 99

                            loss_reg_yaw = tower_loss_mse(yaw_predicted,
                                            label_yaw_cont_placeholder[gpu_index * FLAGS.batch_size:(gpu_index + 1) * FLAGS.batch_size])
                            loss_reg_pitch = tower_loss_mse(pitch_predicted,
                                            label_pitch_cont_placeholder[gpu_index * FLAGS.batch_size:(gpu_index + 1) * FLAGS.batch_size])
                            loss_reg_roll = tower_loss_mse(roll_predicted,
                                            label_roll_cont_placeholder[gpu_index * FLAGS.batch_size:(gpu_index + 1) * FLAGS.batch_size])

                            loss_yaw += FLAGS.alpha * loss_reg_yaw
                            loss_pitch += FLAGS.alpha * loss_reg_pitch
                            loss_roll += FLAGS.alpha * loss_reg_roll

                            loss_utility = loss_yaw + loss_pitch + loss_roll
                            losses_utility.append(loss_utility)
                            losses_yaw.append(loss_yaw)
                            losses_pitch.append(loss_pitch)
                            losses_roll.append(loss_roll)

                            logits_budget = tf.zeros([FLAGS.batch_size, FLAGS.num_classes_budget])
                            loss_budget = 0.0
                            loss_budget_uniform = 0.0
                            weighted_loss_budget_uniform = 0.0
                            loss_uniform_tensor_lst = []
                            for multiplier in multiplier_lst:
                                logits = mobilenet(images_degraded, istraining_placeholder, depth_multiplier=multiplier)
                                loss = tower_loss_xentropy_sparse(logits, budget_labels_placeholder[gpu_index * FLAGS.batch_size:(gpu_index + 1) * FLAGS.batch_size])
                                loss_uniform = tower_loss_xentropy_dense(
                                    logits,
                                    budget_uniform_labels_placeholder[gpu_index * FLAGS.batch_size:(gpu_index + 1) * FLAGS.batch_size, :]
                                )
                                loss_uniform_tensor = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                                labels=budget_uniform_labels_placeholder[gpu_index * FLAGS.batch_size:(gpu_index + 1) * FLAGS.batch_size,:])
                                print(multiplier)
                                print(loss_uniform_tensor)
                                print('##################################################################')
                                logits_budget_lst_dct['{}'.format(multiplier)].append(logits)
                                loss_budget_lst_dct['{}'.format(multiplier)].append(loss)
                                logits_budget += logits
                                loss_budget += loss
                                loss_uniform_tensor_lst.append(loss_uniform_tensor)
                                weighted_loss_budget_uniform += wplaceholder_dct['{}'.format(multiplier)] * loss_uniform
                                loss_budget_uniform += loss_uniform
                            loss_budget_uniform_tensor_stack = tf.stack(loss_uniform_tensor_lst, axis=0)
                            print('##############################################################')
                            print(loss_budget_uniform_tensor_stack.shape)
                            print(tf.reduce_max(loss_budget_uniform_tensor_stack, axis=0).shape)
                            print('##############################################################')
                            argmax_centpy = tf.argmax(loss_budget_uniform_tensor_stack, axis=0)
                            min_centpy = tf.reduce_mean(tf.reduce_max(loss_budget_uniform_tensor_stack, axis=0))
                            logits_budget_lst.append(logits_budget)
                            losses_budget.append(loss_budget)
                            min_centpy_lst.append(min_centpy)
                            argmax_centpy_lst.append(argmax_centpy)
                            loss_budget_uniform_tensor_lst.append(loss_budget_uniform_tensor_stack)

                            if FLAGS.mode == 'SuppressingMostConfident':
                                if FLAGS.use_l1_loss:
                                    loss_degrad = loss_utility + FLAGS._gamma * min_centpy + _lambda_op * L1_loss
                                else:
                                    loss_degrad = loss_utility + FLAGS._gamma * min_centpy
                            elif FLAGS.mode == 'Batch':
                                loss_degrad = loss_utility + FLAGS._gamma * loss_budget_uniform
                            elif FLAGS.mode == 'Online':
                                loss_degrad = loss_utility + FLAGS._gamma * weighted_loss_budget_uniform
                            else:
                                raise ValueError("Wrong given mode")
                            losses_degrad.append(loss_degrad)

                            varlist_degradtion = [v for v in tf.trainable_variables() if any(x in v.name for x in ["DegradationModule"])]
                            print("####################################################DegradationModuleVariables####################################################")
                            print([v.name for v in varlist_degradtion])
                            varlist_utility = [v for v in tf.trainable_variables() if any(x in v.name for x in ["UtilityModule"])]
                            print("####################################################UtilityModuleVariables####################################################")
                            print([v.name for v in varlist_utility])
                            varlist_budget = [v for v in tf.trainable_variables() if any(x in v.name for x in ["BudgetModule"])]
                            print("####################################################BudgetModuleVariables####################################################")
                            print([v.name for v in varlist_budget])

                            grads_degradation = opt_degradation.compute_gradients(loss_degrad, varlist_degradtion)
                            grads_budget = opt_budget.compute_gradients(loss_budget, varlist_budget)
                            grads_utility = opt_utility.compute_gradients(loss_utility, varlist_utility+varlist_degradtion)

                            tower_grads_degradation.append(grads_degradation)
                            tower_grads_budget.append(grads_budget)
                            tower_grads_utility.append(grads_utility)

                            # Reuse variables for the next tower.
                            tf.get_variable_scope().reuse_variables()

            argmax_cent_op = tf.concat(argmax_centpy_lst, 0)

            loss_op = tf.reduce_mean(L2_losses, name='mse')
            psnr_op = tf.multiply(tf.constant(20, dtype=tf.float32),
                              tf.log(1 / tf.sqrt(loss_op)) / tf.log(tf.constant(10, dtype=tf.float32)), name='psnr')

            loss_utility_op = tf.reduce_mean(losses_utility, name='loss_utility')
            loss_budget_op = tf.reduce_mean(losses_budget, name='loss_budget')
            loss_degrad_op = tf.reduce_mean(losses_degrad, name='loss_degrad')


            logits_budget = tf.concat(logits_budget_lst, 0)
            acc_budget_op = accuracy(logits_budget, budget_labels_placeholder)

            acc_op_lst = []
            for multiplier in multiplier_lst:
                acc_op = accuracy(tf.concat(logits_budget_lst_dct['{}'.format(multiplier)], 0), budget_labels_placeholder)
                acc_op_lst.append(acc_op)


            grads_degradation = average_gradients(tower_grads_degradation)
            grads_budget = average_gradients(tower_grads_budget)
            grads_utility = average_gradients(tower_grads_utility)

            with tf.device('/cpu:%d' % 0):
                #tvs_degradation = varlist_degradtion+varlist_instance_norm
                tvs_degradation = varlist_degradtion
                accum_vars_degradtion =  [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in tvs_degradation]
                zero_ops_degradation = [tv.assign(tf.zeros_like(tv)) for tv in accum_vars_degradtion]

            with tf.device('/cpu:%d' % 0):
                tvs_budget = varlist_budget
                accum_vars_budget =  [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in tvs_budget]
                zero_ops_budget = [tv.assign(tf.zeros_like(tv)) for tv in accum_vars_budget]

            with tf.device('/cpu:%d' % 0):
                tvs_utility = varlist_utility + varlist_degradtion
                print(tvs_utility)
                print('###########################################################')
                print(grads_utility)
                accum_vars_utility =  [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in tvs_utility]
                zero_ops_utility = [tv.assign(tf.zeros_like(tv)) for tv in accum_vars_utility]

            #zero_ops = zero_ops_degradation + zero_ops_budget + zero_ops_utility
            global_increment = global_step.assign_add(1)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies([tf.group(*update_ops)]):
                accum_ops_degradation = [accum_vars_degradtion[i].assign_add(gv[0] / FLAGS.n_minibatches) for i, gv in
                                         enumerate(grads_degradation)]
                accum_ops_utility = [accum_vars_utility[i].assign_add(gv[0] / FLAGS.n_minibatches) for i, gv in
                                     enumerate(grads_utility)]
                accum_ops_budget = [accum_vars_budget[i].assign_add(gv[0] / FLAGS.n_minibatches) for i, gv in
                                    enumerate(grads_budget)]

            #accum_ops = accum_ops_degradation + accum_ops_utility + accum_ops_budget
            with tf.control_dependencies([global_increment]):
                apply_gradient_op_degradation = opt_degradation.apply_gradients([(accum_vars_degradtion[i].value(), gv[1]) for i, gv in enumerate(grads_degradation)], global_step=None)

            apply_gradient_op_utility = opt_utility.apply_gradients([(accum_vars_utility[i].value(), gv[1]) for i, gv in enumerate(grads_utility)], global_step=None)
            apply_gradient_op_budget = opt_budget.apply_gradients([(accum_vars_budget[i].value(), gv[1]) for i, gv in enumerate(grads_budget)], global_step=None)

            train_files = [os.path.join(FLAGS.train_files_dir, f) for f in
                           os.listdir(FLAGS.train_files_dir) if f.endswith('.tfrecords')]
            val_files = [os.path.join(FLAGS.val_files_dir, f) for f in
                         os.listdir(FLAGS.val_files_dir) if f.endswith('.tfrecords')]

            print(train_files)
            print(val_files)

            tr_images_op, tr_labels_yaw_op, tr_labels_pitch_op, tr_labels_roll_op, \
            tr_labels_yaw_cont_op, tr_labels_pitch_cont_op, tr_labels_roll_cont_op, tr_genders_op = input_data.inputs_images(filenames=train_files,
                                                                  batch_size=FLAGS.batch_size * FLAGS.gpu_num,
                                                                  num_epochs=None,
                                                                  num_threads=FLAGS.num_threads,
                                                                  num_examples_per_epoch=FLAGS.num_examples_per_epoch)

            val_images_op, val_labels_yaw_op, val_labels_pitch_op, val_labels_roll_op, \
            val_labels_yaw_cont_op, val_labels_pitch_cont_op, val_labels_roll_cont_op, val_genders_op = input_data.inputs_images(filenames=val_files,
                                                                  batch_size=FLAGS.batch_size * FLAGS.gpu_num,
                                                                  num_epochs=None,
                                                                  num_threads=FLAGS.num_threads,
                                                                  num_examples_per_epoch=FLAGS.num_examples_per_epoch)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            init_op = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())
            sess.run(init_op)
            # Create a saver for writing training checkpoints.

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

            def restore_model_pretrained_MobileNet(multiplier):
                varlist = [v for v in tf.trainable_variables() if
                           any(x in v.name.split('/')[0] for x in ["BudgetModule_{}".format(multiplier)])]
                varlist = [v for v in varlist if not any(x in v.name for x in ["Conv2d_1c_1x1"])]
                print("###############################BudgetModule_{}###############################".format(multiplier))
                print(varlist)
                vardict = {v.name[:-2].replace('BudgetModule_{}'.format(multiplier), 'MobilenetV1'): v for v in varlist}

                mobilenet_dict = {1.0: FLAGS.pretrained_MobileNet_10,
                                  0.75: FLAGS.pretrained_MobileNet_075,
                                  0.5: FLAGS.pretrained_MobileNet_050,
                                  0.25: FLAGS.pretrained_MobileNet_025,
                                  }
                saver = tf.train.Saver(vardict)
                ckpt = tf.train.get_checkpoint_state(checkpoint_dir=mobilenet_dict[multiplier])
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    print(
                        '#############################Session restored from pretrained model at {}!###############################'.format(
                            ckpt.model_checkpoint_path))
                else:
                    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), mobilenet_dict[multiplier])

            gvar_list = tf.global_variables()
            bn_moving_vars = [g for g in gvar_list if 'moving_mean' in g.name]
            bn_moving_vars += [g for g in gvar_list if 'moving_variance' in g.name]

            if use_whole_pretrained_model:
                varlist = [v for v in tf.trainable_variables() + bn_moving_vars if
                                      any(x in v.name for x in ["DegradationModule", "UtilityModule"])]
                saver = tf.train.Saver(varlist)
                ckpt = tf.train.get_checkpoint_state(checkpoint_dir=FLAGS.whole_pretraining)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    print('Session restored from trained model at {}!'.format(ckpt.model_checkpoint_path))
                else:
                    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), FLAGS.whole_pretraining)
            else:
                varlist = tf.trainable_variables() + bn_moving_vars
                saver = tf.train.Saver(varlist)
                ckpt = tf.train.get_checkpoint_state(checkpoint_dir=FLAGS.checkpoint_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    print('Session restored from trained model at {}!'.format(ckpt.model_checkpoint_path))
                else:
                    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), FLAGS.checkpoint_dir)

            varlist = tf.trainable_variables() + bn_moving_vars
            saver = tf.train.Saver(var_list=varlist, max_to_keep=5)
            #ckpt_saver = tf.train.Saver(var_list=varlist, max_to_keep=20)

            if not os.path.exists(FLAGS.summary_dir):
                os.makedirs(FLAGS.summary_dir)
            loss_summary_file = open(FLAGS.summary_dir+'loss_summary.txt', 'w')
            train_summary_file = open(FLAGS.summary_dir+'train_summary.txt', 'w')
            val_summary_file = open(FLAGS.summary_dir+'val_summary.txt', 'w')
            model_sampling_summary_file = open(FLAGS.summary_dir+'model_summary.txt', 'w')

            budget_uniform_labels = np.full((FLAGS.batch_size * FLAGS.gpu_num, FLAGS.num_classes_budget),
                                              1 / FLAGS.num_classes_budget, dtype=np.float32)
            for step in xrange(FLAGS.max_steps):

                if FLAGS.use_restarting and step % FLAGS.restart_step == 0:
                    budget_varlist = [v for v in tf.trainable_variables() if any(x in v.name for x in ["BudgetModule"])]
                    init_budget_op = tf.variables_initializer(budget_varlist)
                    sess.run(init_budget_op)
                    for _ in itertools.repeat(None, FLAGS.retraining_step):
                        # saver = tf.train.Saver(tf.trainable_variables() + bn_moving_vars)
                        start_time = time.time()
                        acc_budget_lst, loss_degrad_lst, loss_utility_lst, loss_budget_lst, psnr_lst = [], [], [], [], []
                        sess.run([zero_ops_budget])
                        for _ in itertools.repeat(None, FLAGS.n_minibatches):
                            images, labels_yaw, labels_pitch, labels_roll, labels_yaw_cont, \
                            labels_pitch_cont, labels_roll_cont, genders = sess.run([tr_images_op,
                                                                                                 tr_labels_yaw_op,
                                                                                                 tr_labels_pitch_op,
                                                                                                 tr_labels_roll_op,
                                                                                                 tr_labels_yaw_cont_op,
                                                                                                 tr_labels_pitch_cont_op,
                                                                                                 tr_labels_roll_cont_op,
                                                                                                 tr_genders_op])

                            _, psnr, acc_budget, loss_degrad_value, loss_utility_value, loss_budget_value = sess.run(
                                [accum_ops_budget, psnr_op, acc_budget_op, loss_degrad_op,
                                 loss_utility_op, loss_budget_op],
                                feed_dict={images_placeholder: images,
                                           label_yaw_placeholder: labels_yaw,
                                           label_pitch_placeholder: labels_pitch,
                                           label_roll_placeholder: labels_roll,
                                           label_yaw_cont_placeholder: labels_yaw_cont,
                                           label_pitch_cont_placeholder: labels_pitch_cont,
                                           label_roll_cont_placeholder: labels_roll_cont,
                                           budget_uniform_labels_placeholder: budget_uniform_labels,
                                           budget_labels_placeholder: genders,
                                           istraining_placeholder: True})
                            acc_budget_lst.append(acc_budget)
                            psnr_lst.append(psnr)
                            loss_degrad_lst.append(loss_degrad_value)
                            loss_utility_lst.append(loss_utility_value)
                            loss_budget_lst.append(loss_budget_value)
                        sess.run(apply_gradient_op_budget)
                        assert not np.isnan(np.mean(loss_degrad_lst)), 'Model diverged with loss = NaN'
                        loss_summary = 'Restarting (Budget), Step: {:4d}, time: {:.4f}, budget loss: {:.8f}, training budget accuracy: {:.5f}, utility loss: {:.8f}, psnr: {:.8f}'.format(step,
                                        time.time() - start_time, np.mean(loss_budget_lst), np.mean(acc_budget_lst), np.mean(loss_utility_lst), np.mean(psnr_lst))
                        model_sampling_summary_file.write(loss_summary + '\n')
                        print(loss_summary)

                start_time = time.time()
                loss_degrad_lst, loss_utility_lst, loss_budget_lst, psnr_lst = [], [], [], []
                sess.run(zero_ops_degradation)
                for _ in itertools.repeat(None, FLAGS.n_minibatches):
                    images, labels_yaw, labels_pitch, labels_roll, labels_yaw_cont, \
                    labels_pitch_cont, labels_roll_cont, genders = sess.run([tr_images_op,
                                                                    tr_labels_yaw_op, tr_labels_pitch_op,tr_labels_roll_op,
                                                                    tr_labels_yaw_cont_op, tr_labels_pitch_cont_op, tr_labels_roll_cont_op,
                                                                    tr_genders_op])

                    _, psnr, argmax_cent, loss_degrad_value, loss_utility_value, loss_budget_value = sess.run([accum_ops_degradation, psnr_op,
                                                        argmax_cent_op, loss_degrad_op, loss_utility_op, loss_budget_op],
                                                    feed_dict={images_placeholder: images,
                                                               label_yaw_placeholder: labels_yaw,
                                                               label_pitch_placeholder: labels_pitch,
                                                               label_roll_placeholder: labels_roll,
                                                               label_yaw_cont_placeholder: labels_yaw_cont,
                                                               label_pitch_cont_placeholder: labels_pitch_cont,
                                                               label_roll_cont_placeholder: labels_roll_cont,
                                                               budget_uniform_labels_placeholder: budget_uniform_labels,
                                                               budget_labels_placeholder: genders,
                                                               istraining_placeholder: True,
                                                               })
                    print(argmax_cent)
                    #print(loss_budget_uniform_tensor)
                    psnr_lst.append(psnr)
                    loss_degrad_lst.append(loss_degrad_value)
                    loss_utility_lst.append(loss_utility_value)
                    loss_budget_lst.append(loss_budget_value)
                _, _lambda = sess.run([apply_gradient_op_degradation, _lambda_op])

                assert not np.isnan(np.mean(loss_degrad_lst)), 'Model diverged with loss = NaN'
                duration = time.time() - start_time


                loss_summary = 'Alternating Training (Degradation), Lambda: {:.8f}, Step: {:4d}, time: {:.4f}, degradation loss: {:.8f}, utility loss: {:.8f}, budget loss: {:.8f}, psnr: {:.8f}'.format(_lambda, step,
                                                        duration, np.mean(loss_degrad_lst), np.mean(loss_utility_lst), np.mean(loss_budget_lst), np.mean(psnr_lst))

                print(loss_summary)
                loss_summary_file.write(loss_summary + '\n')

                if FLAGS.use_monitor_utility:
                    while True:
                    #for _ in itertools.repeat(None, FLAGS.adaptation_utility_steps):
                        start_time = time.time()
                        acc_budget_lst, loss_degrad_lst, loss_utility_lst, loss_budget_lst, psnr_lst = [], [], [], [], []
                        for _ in itertools.repeat(None, FLAGS.n_minibatches):
                            images, labels_yaw, labels_pitch, labels_roll, labels_yaw_cont, \
                            labels_pitch_cont, labels_roll_cont, genders = sess.run([val_images_op,
                                                                                                 val_labels_yaw_op,
                                                                                                 val_labels_pitch_op,
                                                                                                 val_labels_roll_op,
                                                                                                 val_labels_yaw_cont_op,
                                                                                                 val_labels_pitch_cont_op,
                                                                                                 val_labels_roll_cont_op,
                                                                                                 val_genders_op])

                            acc_budget, psnr, loss_degrad, loss_utility, loss_budget = sess.run(
                                                        [acc_budget_op, psnr_op, loss_degrad_op, loss_utility_op, loss_budget_op],
                                                        feed_dict={images_placeholder: images,
                                                                   label_yaw_placeholder: labels_yaw,
                                                                   label_pitch_placeholder: labels_pitch,
                                                                   label_roll_placeholder: labels_roll,
                                                                   label_yaw_cont_placeholder: labels_yaw_cont,
                                                                   label_pitch_cont_placeholder: labels_pitch_cont,
                                                                   label_roll_cont_placeholder: labels_roll_cont,
                                                                   budget_uniform_labels_placeholder: budget_uniform_labels,
                                                                   budget_labels_placeholder: genders,
                                                                   istraining_placeholder: True,
                                                        })
                            acc_budget_lst.append(acc_budget)
                            psnr_lst.append(psnr)
                            loss_degrad_lst.append(loss_degrad)
                            loss_utility_lst.append(loss_utility)
                            loss_budget_lst.append(loss_budget)
                            # test_writer.add_summary(summary, step)
                        val_summary = "Step: {:4d}, time: {:.4f}, degradation loss: {:.8f}, utility loss: {:.8f}, budget loss: {:.8f}, validation budget accuracy: {:.5f}, psnr: {:.8f}" .format(
                                step,
                                time.time() - start_time, np.mean(loss_degrad_lst),
                                np.mean(loss_utility_lst), np.mean(loss_budget_lst),
                                np.mean(acc_budget_lst), np.mean(psnr_lst))
                        print(val_summary)

                        if np.mean(loss_utility_lst) <= FLAGS.util_loss_val_thresh:
                            break
                        start_time = time.time()
                        sess.run(zero_ops_utility)
                        acc_budget_lst, loss_degrad_lst, loss_utility_lst, loss_budget_lst, psnr_lst = [], [], [], [], []
                        for _ in itertools.repeat(None, FLAGS.n_minibatches):
                            images, labels_yaw, labels_pitch, labels_roll, labels_yaw_cont, \
                            labels_pitch_cont, labels_roll_cont, genders = sess.run([val_images_op,
                                                                                                 val_labels_yaw_op,
                                                                                                 val_labels_pitch_op,
                                                                                                 val_labels_roll_op,
                                                                                                 val_labels_yaw_cont_op,
                                                                                                 val_labels_pitch_cont_op,
                                                                                                 val_labels_roll_cont_op,
                                                                                                 val_genders_op])

                            _, psnr, acc_budget, loss_degrad, loss_utility, loss_budget = sess.run([accum_ops_utility, psnr_op,
                                                            acc_budget_op, loss_degrad_op, loss_utility_op, loss_budget_op],
                                                            feed_dict={images_placeholder: images,
                                                                       label_yaw_placeholder: labels_yaw,
                                                                       label_pitch_placeholder: labels_pitch,
                                                                       label_roll_placeholder: labels_roll,
                                                                       label_yaw_cont_placeholder: labels_yaw_cont,
                                                                       label_pitch_cont_placeholder: labels_pitch_cont,
                                                                       label_roll_cont_placeholder: labels_roll_cont,
                                                                       budget_uniform_labels_placeholder: budget_uniform_labels,
                                                                       budget_labels_placeholder: genders,
                                                                       istraining_placeholder: True,
                                                               })
                            acc_budget_lst.append(acc_budget)
                            psnr_lst.append(psnr)
                            loss_degrad_lst.append(loss_degrad)
                            loss_utility_lst.append(loss_utility)
                            loss_budget_lst.append(loss_budget)
                        sess.run([apply_gradient_op_utility])
                        assert not np.isnan(np.mean(loss_degrad_lst)), 'Model diverged with loss = NaN'
                        loss_summary = 'Alternating Training (Utility), Step: {:4d}, time: {:.4f}, degradation loss: {:.8f}, utility loss: {:.8f}, budget loss: {:.8f}, psnr: {:.8f}'.format(step,
                                                            time.time() - start_time, np.mean(loss_degrad_lst),
                                                            np.mean(loss_utility_lst), np.mean(loss_budget_lst), np.mean(psnr_lst))

                        print(loss_summary)

                if FLAGS.use_monitor_budget:
                    while True:
                        start_time = time.time()
                        sess.run(zero_ops_budget)
                        acc_budget_lst, loss_degrad_lst, loss_utility_lst, loss_budget_lst, psnr_lst = [], [], [], [], []
                        for _ in itertools.repeat(None, FLAGS.n_minibatches):
                            images, labels_yaw, labels_pitch, labels_roll, labels_yaw_cont, \
                            labels_pitch_cont, labels_roll_cont, genders = sess.run([tr_images_op,
                                                                                                 tr_labels_yaw_op,
                                                                                                 tr_labels_pitch_op,
                                                                                                 tr_labels_roll_op,
                                                                                                 tr_labels_yaw_cont_op,
                                                                                                 tr_labels_pitch_cont_op,
                                                                                                 tr_labels_roll_cont_op,
                                                                                                 tr_genders_op])

                            _, psnr, acc_budget, loss_degrad, loss_utility, loss_budget = sess.run([accum_ops_budget, psnr_op, acc_budget_op, loss_degrad_op, loss_utility_op, loss_budget_op],
                                                    feed_dict={images_placeholder: images,
                                                                   label_yaw_placeholder: labels_yaw,
                                                                   label_pitch_placeholder: labels_pitch,
                                                                   label_roll_placeholder: labels_roll,
                                                                   label_yaw_cont_placeholder: labels_yaw_cont,
                                                                   label_pitch_cont_placeholder: labels_pitch_cont,
                                                                   label_roll_cont_placeholder: labels_roll_cont,
                                                                   budget_uniform_labels_placeholder: budget_uniform_labels,
                                                                   budget_labels_placeholder: genders,
                                                                   istraining_placeholder: True,
                                                               })
                            acc_budget_lst.append(acc_budget)
                            psnr_lst.append(psnr)
                            loss_degrad_lst.append(loss_degrad)
                            loss_utility_lst.append(loss_utility)
                            loss_budget_lst.append(loss_budget)
                        sess.run([apply_gradient_op_budget])
                        assert not np.isnan(np.mean(loss_degrad_lst)), 'Model diverged with loss = NaN'
                        loss_summary = 'Alternating Training (Budget), Step: {:4d}, time: {:.4f}, degradation loss: {:.8f}, utility loss: {:.8f}, budget loss: {:.8f}, ' \
                                   ' budget accuracy: {:.5f}, psnr: {:.8f}'.format(step,
                                                            time.time() - start_time, np.mean(loss_degrad_lst),
                                                            np.mean(loss_utility_lst), np.mean(loss_budget_lst),
                                                            np.mean(acc_budget_lst), np.mean(psnr_lst))

                        print(loss_summary)
                        if np.mean(acc_budget_lst) >= FLAGS.budget_acc_train_thresh:
                            break

                if step % FLAGS.val_step == 0:
                    start_time = time.time()
                    acc_budget_lst, loss_degrad_lst, loss_utility_lst, loss_budget_lst, psnr_lst = [], [], [], [], []
                    for _ in itertools.repeat(None, 20):
                        images, labels_yaw, labels_pitch, labels_roll, labels_yaw_cont, \
                        labels_pitch_cont, labels_roll_cont, genders = sess.run([tr_images_op,
                                                                                             tr_labels_yaw_op,
                                                                                             tr_labels_pitch_op,
                                                                                             tr_labels_roll_op,
                                                                                             tr_labels_yaw_cont_op,
                                                                                             tr_labels_pitch_cont_op,
                                                                                             tr_labels_roll_cont_op,
                                                                                             tr_genders_op])

                        acc_budget, psnr, loss_degrad, loss_utility, loss_budget = sess.run(
                                                        [acc_budget_op, psnr_op, loss_degrad_op, loss_utility_op, loss_budget_op],
                                                        feed_dict={images_placeholder: images,
                                                                   label_yaw_placeholder: labels_yaw,
                                                                   label_pitch_placeholder: labels_pitch,
                                                                   label_roll_placeholder: labels_roll,
                                                                   label_yaw_cont_placeholder: labels_yaw_cont,
                                                                   label_pitch_cont_placeholder: labels_pitch_cont,
                                                                   label_roll_cont_placeholder: labels_roll_cont,
                                                                   budget_uniform_labels_placeholder: budget_uniform_labels,
                                                                   budget_labels_placeholder: genders,
                                                                   istraining_placeholder: True,
                                                                   })
                        acc_budget_lst.append(acc_budget)
                        psnr_lst.append(psnr)
                        loss_degrad_lst.append(loss_degrad)
                        loss_utility_lst.append(loss_utility)
                        loss_budget_lst.append(loss_budget)

                    train_summary = "Step: {:4d}, time: {:.4f}, degradation loss: {:.8f}, training utility loss: {:.8f}, training budget loss: {:.8f}, training budget accuracy: {:.5f}, psnr: {:.8f}".format(
                                    step, time.time() - start_time, np.mean(loss_degrad_lst),
                                    np.mean(loss_utility_lst), np.mean(loss_budget_lst), np.mean(acc_budget_lst), np.mean(psnr_lst))
                    print(train_summary)
                    train_summary_file.write(train_summary + '\n')

                    start_time = time.time()
                    acc_budget_lst, loss_degrad_lst, loss_utility_lst, loss_budget_lst, psnr_lst = [], [], [], [], []
                    for _ in itertools.repeat(None, 20):
                        images, labels_yaw, labels_pitch, labels_roll, labels_yaw_cont, \
                        labels_pitch_cont, labels_roll_cont, genders = sess.run([val_images_op,
                                                                                             val_labels_yaw_op,
                                                                                             val_labels_pitch_op,
                                                                                             val_labels_roll_op,
                                                                                             val_labels_yaw_cont_op,
                                                                                             val_labels_pitch_cont_op,
                                                                                             val_labels_roll_cont_op,
                                                                                             val_genders_op])

                        acc_budget, psnr, loss_degrad, loss_utility, loss_budget = sess.run(
                                                        [acc_budget_op, psnr_op, loss_degrad_op, loss_utility_op, loss_budget_op],
                                                        feed_dict={images_placeholder: images,
                                                                   label_yaw_placeholder: labels_yaw,
                                                                   label_pitch_placeholder: labels_pitch,
                                                                   label_roll_placeholder: labels_roll,
                                                                   label_yaw_cont_placeholder: labels_yaw_cont,
                                                                   label_pitch_cont_placeholder: labels_pitch_cont,
                                                                   label_roll_cont_placeholder: labels_roll_cont,
                                                                   budget_uniform_labels_placeholder: budget_uniform_labels,
                                                                   budget_labels_placeholder: genders,
                                                                   istraining_placeholder: True,
                                                                   })
                        acc_budget_lst.append(acc_budget)
                        psnr_lst.append(psnr)
                        loss_degrad_lst.append(loss_degrad)
                        loss_utility_lst.append(loss_utility)
                        loss_budget_lst.append(loss_budget)
                        # test_writer.add_summary(summary, step)
                    val_summary = "Step: {:4d}, time: {:.4f}, degradation loss: {:.8f}, validation utility loss: {:.8f}, validation budget loss: {:.8f}, validation budget accuracy: {:.5f}, psnr: {:.8f}".format(
                                    step, time.time() - start_time, np.mean(loss_degrad_lst),
                                    np.mean(loss_utility_lst), np.mean(loss_budget_lst), np.mean(acc_budget_lst), np.mean(psnr_lst))
                    print(val_summary)
                    val_summary_file.write(val_summary + '\n')

                if step % FLAGS.save_step == 0 or (step + 1) == FLAGS.max_steps:
                    checkpoint_path = os.path.join(FLAGS.checkpoint_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)

            loss_summary_file.close()
            train_summary_file.close()
            val_summary_file.close()
            coord.request_stop()
            coord.join(threads)

    print("done")

def run_training_multi_model_mixed_restarting():
    # Create model directory
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    #if not os.path.exists(FLAGS.saved_checkpoint_dir):
    #    os.makedirs(FLAGS.saved_checkpoint_dir)

    use_whole_pretrained_model = True
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True

    #multiplier_lst = [0.5 - i * 0.02 for i in range(FLAGS.NBudget)]
    from collections import defaultdict
    logits_budget_lst_dct = defaultdict(list)
    loss_budget_lst_dct = defaultdict(list)
    wplaceholder_dct = {}

    with tf.Graph().as_default():
        with tf.Session(config=config) as sess:
            global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
            images_placeholder, label_yaw_placeholder, label_pitch_placeholder, label_roll_placeholder, label_yaw_cont_placeholder, label_pitch_cont_placeholder, label_roll_cont_placeholder = placeholder_inputs(
                FLAGS.batch_size * FLAGS.gpu_num)
            budget_labels_placeholder =  tf.placeholder(tf.int64, shape=(FLAGS.batch_size * FLAGS.gpu_num))

            istraining_placeholder = tf.placeholder(tf.bool)
            budgetNet_model_name_lst = ['mobilenet_v1_050', 'resnet_v2_50', 'mobilenet_v1', 'mobilenet_v1_075']
            budgetNet_dict = {}
            for model_name in budgetNet_model_name_lst:
                wplaceholder_dct['{}'.format(model_name)] = tf.placeholder(tf.float32)
                budgetNet_dict[model_name] = nets_factory.get_network_fn(model_name,
                                                                        num_classes=FLAGS.num_classes_budget,
                                                                        weight_decay=FLAGS.weight_decay,
                                                                        is_training=istraining_placeholder)
            L2_losses = []
            losses_utility = []
            losses_yaw = []
            losses_pitch = []
            losses_roll = []

            tower_grads_degradation = []
            tower_grads_utility = []
            tower_grads_budget = []

            # Compute Acc
            logits_budget_lst = []

            # Compute Loss
            losses_budget = []
            losses_degrad = []

            # Compute prediction with min entropy (most confident)
            # Use max uniform loss instead
            argmax_nentropy_lst = []
            opt_degradation = tf.train.AdamOptimizer(FLAGS.degradation_lr)
            opt_utility = tf.train.AdamOptimizer(FLAGS.utility_lr)
            opt_budget = tf.train.AdamOptimizer(FLAGS.budget_lr)

            if FLAGS.use_lambda_decay:
                _lambda_op = tf.train.exponential_decay(FLAGS._lambda, global_step=global_step, decay_steps=10, decay_rate=0.9)
            else:
                _lambda_op = tf.identity(FLAGS._lambda)
            with tf.variable_scope(tf.get_variable_scope()) as scope:
                for gpu_index in range(0, FLAGS.gpu_num):
                    with tf.device('/gpu:%d' % gpu_index):
                        print('/gpu:%d' % gpu_index)
                        with tf.name_scope('%s_%d' % ('gpu', gpu_index)) as scope:
                            images_degraded = residualnet(images_placeholder[gpu_index * FLAGS.batch_size:(gpu_index + 1) * FLAGS.batch_size])
                            L2_loss = tower_loss_mse(images_degraded, images_placeholder[gpu_index * FLAGS.batch_size:(gpu_index + 1) * FLAGS.batch_size])
                            L2_losses.append(L2_loss)

                            L1_loss = tf.reduce_mean(tf.abs(images_degraded - images_placeholder[gpu_index * FLAGS.batch_size:(gpu_index + 1) * FLAGS.batch_size]))

                            if FLAGS.use_avg_replicate:
                                images_degraded = _avg_replicate(images_degraded)

                            fc_yaw, fc_pitch, fc_roll = hopenet(images_degraded, is_training=istraining_placeholder)
                            loss_yaw = tower_loss_xentropy_sparse(fc_yaw,
                                                                  label_yaw_placeholder[gpu_index * FLAGS.batch_size:(gpu_index + 1) * FLAGS.batch_size])
                            loss_pitch = tower_loss_xentropy_sparse(fc_pitch,
                                                                  label_pitch_placeholder[gpu_index * FLAGS.batch_size:(gpu_index + 1) * FLAGS.batch_size])
                            loss_roll = tower_loss_xentropy_sparse(fc_roll,
                                                                   label_roll_placeholder[gpu_index * FLAGS.batch_size:(gpu_index + 1) * FLAGS.batch_size])

                            idx_tensor = tf.expand_dims(tf.range(66, dtype=tf.float32), 0)
                            idx_tensor = tf.tile(idx_tensor, [FLAGS.batch_size, 1])

                            yaw_predicted = tf.reduce_sum(tf.multiply(tf.nn.softmax(fc_yaw, axis=1), idx_tensor),
                                                          axis=1) * 3 - 99
                            pitch_predicted = tf.reduce_sum(tf.multiply(tf.nn.softmax(fc_pitch, axis=1), idx_tensor),
                                                            axis=1) * 3 - 99
                            roll_predicted = tf.reduce_sum(tf.multiply(tf.nn.softmax(fc_roll, axis=1), idx_tensor),
                                                           axis=1) * 3 - 99

                            loss_reg_yaw = tower_loss_mse(yaw_predicted,
                                            label_yaw_cont_placeholder[gpu_index * FLAGS.batch_size:(gpu_index + 1) * FLAGS.batch_size])
                            loss_reg_pitch = tower_loss_mse(pitch_predicted,
                                            label_pitch_cont_placeholder[gpu_index * FLAGS.batch_size:(gpu_index + 1) * FLAGS.batch_size])
                            loss_reg_roll = tower_loss_mse(roll_predicted,
                                            label_roll_cont_placeholder[gpu_index * FLAGS.batch_size:(gpu_index + 1) * FLAGS.batch_size])

                            loss_yaw += FLAGS.alpha * loss_reg_yaw
                            loss_pitch += FLAGS.alpha * loss_reg_pitch
                            loss_roll += FLAGS.alpha * loss_reg_roll

                            loss_utility = loss_yaw + loss_pitch + loss_roll
                            losses_utility.append(loss_utility)
                            losses_yaw.append(loss_yaw)
                            losses_pitch.append(loss_pitch)
                            losses_roll.append(loss_roll)

                            logits_budget = tf.zeros([FLAGS.batch_size, FLAGS.num_classes_budget])
                            loss_budget = 0.0
                            loss_budget_nentropy = 0.0
                            weighted_loss_budget_nentropy = 0.0
                            logits_lst = []
                            for model_name in budgetNet_model_name_lst:
                                print(model_name)
                                print(tf.trainable_variables())
                                logits, _ = budgetNet_dict[model_name](images_degraded)
                                logits_lst.append(logits)
                                loss = tower_loss_xentropy_sparse(logits, budget_labels_placeholder[gpu_index * FLAGS.batch_size:(gpu_index + 1) * FLAGS.batch_size])
                                loss_neg_entropy = tower_loss_neg_entropy(logits)
                                logits_budget_lst_dct['{}'.format(model_name)].append(logits)
                                loss_budget_lst_dct['{}'.format(model_name)].append(loss)
                                logits_budget += logits / len(budgetNet_model_name_lst)
                                loss_budget += loss / len(budgetNet_model_name_lst)

                                weighted_loss_budget_nentropy += wplaceholder_dct['{}'.format(model_name)] * loss_neg_entropy
                                loss_budget_nentropy += loss_neg_entropy / len(budgetNet_model_name_lst)

                            max_nentropy, argmax_nentropy = tower_loss_max_neg_entropy(logits_lst)
                            logits_budget_lst.append(logits_budget)
                            losses_budget.append(loss_budget)
                            print(argmax_nentropy.shape)
                            argmax_nentropy_lst.append(argmax_nentropy)

                            if FLAGS.mode == 'SuppressingMostConfident':
                                if FLAGS.use_l1_loss:
                                    loss_degrad = loss_utility + FLAGS._gamma * max_nentropy + _lambda_op * L1_loss
                                else:
                                    loss_degrad = loss_utility + FLAGS._gamma * max_nentropy
                            elif FLAGS.mode == 'Batch':
                                loss_degrad = loss_utility + FLAGS._gamma * loss_budget_nentropy
                            elif FLAGS.mode == 'Online':
                                loss_degrad = loss_utility + FLAGS._gamma * weighted_loss_budget_nentropy
                            else:
                                raise ValueError("Wrong given mode")
                            losses_degrad.append(loss_degrad)

                            varlist_degradtion = [v for v in tf.trainable_variables() if any(x in v.name for x in ["DegradationModule"])]
                            print("####################################################DegradationModuleVariables####################################################")
                            print([v.name for v in varlist_degradtion])
                            varlist_utility = [v for v in tf.trainable_variables() if any(x in v.name for x in ["UtilityModule"])]
                            print("####################################################UtilityModuleVariables####################################################")
                            print([v.name for v in varlist_utility])
                            varlist_budget = [v for v in tf.trainable_variables() if not any(x in v.name for x in ["DegradationModule", "UtilityModule"])]
                            print("####################################################BudgetModuleVariables####################################################")
                            print([v.name for v in varlist_budget])

                            grads_degradation = opt_degradation.compute_gradients(loss_degrad, varlist_degradtion)
                            grads_budget = opt_budget.compute_gradients(loss_budget, varlist_budget)
                            grads_utility = opt_utility.compute_gradients(loss_utility, varlist_utility+varlist_degradtion)

                            tower_grads_degradation.append(grads_degradation)
                            tower_grads_budget.append(grads_budget)
                            tower_grads_utility.append(grads_utility)

                            # Reuse variables for the next tower.
                            tf.get_variable_scope().reuse_variables()

            argmax_nentropy_op = tf.concat(argmax_nentropy_lst, 0)

            L2_loss = tf.reduce_mean(L2_losses, name='mse')
            psnr_op = tf.multiply(tf.constant(20, dtype=tf.float32),
                              tf.log(1 / tf.sqrt(L2_loss)) / tf.log(tf.constant(10, dtype=tf.float32)), name='psnr')

            loss_utility_op = tf.reduce_mean(losses_utility, name='loss_utility')
            loss_budget_op = tf.reduce_mean(losses_budget, name='loss_budget')
            loss_degrad_op = tf.reduce_mean(losses_degrad, name='loss_degrad')


            logits_budget = tf.concat(logits_budget_lst, 0)
            acc_budget_ensemble_op = accuracy(logits_budget, budget_labels_placeholder)

            acc_budget_op_lst = []
            for model_name in budgetNet_model_name_lst:
                acc_budget_op = accuracy(tf.concat(logits_budget_lst_dct['{}'.format(model_name)], 0), budget_labels_placeholder)
                acc_budget_op_lst.append(acc_budget_op)


            grads_degradation = average_gradients(tower_grads_degradation)
            grads_budget = average_gradients(tower_grads_budget)
            grads_utility = average_gradients(tower_grads_utility)

            with tf.device('/cpu:%d' % 0):
                #tvs_degradation = varlist_degradtion+varlist_instance_norm
                tvs_degradation = varlist_degradtion
                accum_vars_degradtion =  [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in tvs_degradation]
                zero_ops_degradation = [tv.assign(tf.zeros_like(tv)) for tv in accum_vars_degradtion]

            with tf.device('/cpu:%d' % 0):
                tvs_budget = varlist_budget
                accum_vars_budget =  [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in tvs_budget]
                zero_ops_budget = [tv.assign(tf.zeros_like(tv)) for tv in accum_vars_budget]

            with tf.device('/cpu:%d' % 0):
                tvs_utility = varlist_utility + varlist_degradtion
                print(tvs_utility)
                print('###########################################################')
                print(grads_utility)
                accum_vars_utility =  [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in tvs_utility]
                zero_ops_utility = [tv.assign(tf.zeros_like(tv)) for tv in accum_vars_utility]

            #zero_ops = zero_ops_degradation + zero_ops_budget + zero_ops_utility
            global_increment = global_step.assign_add(1)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies([tf.group(*update_ops)]):
                accum_ops_degradation = [accum_vars_degradtion[i].assign_add(gv[0] / FLAGS.n_minibatches) for i, gv in
                                         enumerate(grads_degradation)]
                accum_ops_utility = [accum_vars_utility[i].assign_add(gv[0] / FLAGS.n_minibatches) for i, gv in
                                     enumerate(grads_utility)]
                accum_ops_budget = [accum_vars_budget[i].assign_add(gv[0] / FLAGS.n_minibatches) for i, gv in
                                    enumerate(grads_budget)]

            #accum_ops = accum_ops_degradation + accum_ops_utility + accum_ops_budget
            with tf.control_dependencies([global_increment]):
                apply_gradient_op_degradation = opt_degradation.apply_gradients([(accum_vars_degradtion[i].value(), gv[1]) for i, gv in enumerate(grads_degradation)], global_step=None)

            apply_gradient_op_utility = opt_utility.apply_gradients([(accum_vars_utility[i].value(), gv[1]) for i, gv in enumerate(grads_utility)], global_step=None)
            apply_gradient_op_budget = opt_budget.apply_gradients([(accum_vars_budget[i].value(), gv[1]) for i, gv in enumerate(grads_budget)], global_step=None)

            train_files = [os.path.join(FLAGS.train_files_dir, f) for f in
                           os.listdir(FLAGS.train_files_dir) if f.endswith('.tfrecords')]
            val_files = [os.path.join(FLAGS.val_files_dir, f) for f in
                         os.listdir(FLAGS.val_files_dir) if f.endswith('.tfrecords')]

            print(train_files)
            print(val_files)

            tr_images_op, tr_labels_yaw_op, tr_labels_pitch_op, tr_labels_roll_op, \
            tr_labels_yaw_cont_op, tr_labels_pitch_cont_op, tr_labels_roll_cont_op, tr_genders_op = input_data.inputs_images(filenames=train_files,
                                                                  batch_size=FLAGS.batch_size * FLAGS.gpu_num,
                                                                  num_epochs=None,
                                                                  num_threads=FLAGS.num_threads,
                                                                  num_examples_per_epoch=FLAGS.num_examples_per_epoch)

            val_images_op, val_labels_yaw_op, val_labels_pitch_op, val_labels_roll_op, \
            val_labels_yaw_cont_op, val_labels_pitch_cont_op, val_labels_roll_cont_op, val_genders_op = input_data.inputs_images(filenames=val_files,
                                                                  batch_size=FLAGS.batch_size * FLAGS.gpu_num,
                                                                  num_epochs=None,
                                                                  num_threads=FLAGS.num_threads,
                                                                  num_examples_per_epoch=FLAGS.num_examples_per_epoch)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            init_op = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())
            sess.run(init_op)
            # Create a saver for writing training checkpoints.


            def restore_model(dir, varlist, modulename):
                import re
                regex = re.compile(r'(MobilenetV1_)(\d+\.\d+)', re.IGNORECASE)
                if 'mobilenet' in modulename:
                    varlist = {regex.sub('MobilenetV1', v.name[:-2]): v for v in varlist}
                if os.path.isfile(dir):
                    print(varlist)
                    saver = tf.train.Saver(varlist)
                    saver.restore(sess, dir)
                    print('#############################Session restored from pretrained model at {}!#############################'.format(dir))
                else:
                    ckpt = tf.train.get_checkpoint_state(checkpoint_dir=dir)
                    if ckpt and ckpt.model_checkpoint_path:
                        saver = tf.train.Saver(varlist)
                        saver.restore(sess, ckpt.model_checkpoint_path)
                        print('#############################Session restored from pretrained model at {}!#############################'.format(
                            ckpt.model_checkpoint_path))


            gvar_list = tf.global_variables()
            bn_moving_vars = [g for g in gvar_list if 'moving_mean' in g.name]
            bn_moving_vars += [g for g in gvar_list if 'moving_variance' in g.name]

            def model_restarting_from_pretrained():
                varlist_dict = {}
                pretrained_model_dir = '/home/wuzhenyu_sjtu/DAN_AFLW/checkpoint/pretrained_budget/{}'
                for model_name in budgetNet_model_name_lst:
                    print(model_name)
                    if model_name in model_name_mapping:
                        varlist = [v for v in tf.trainable_variables() if any(x in v.name for x in [model_name_mapping[model_name]])]
                    else:
                        varlist = [v for v in tf.trainable_variables() if any(x in v.name for x in [model_name])]
                    varlist_dict[model_name] = [v for v in varlist if not any(x in v.name for x in ["logits"])]
                    print(ckpt_path_map[model_name])
                    restore_model(pretrained_model_dir.format(ckpt_path_map[model_name]), varlist_dict[model_name], model_name)

            if use_whole_pretrained_model:
                varlist = [v for v in tf.trainable_variables() + bn_moving_vars if
                                      any(x in v.name for x in ["DegradationModule", "UtilityModule"])]
                saver = tf.train.Saver(varlist)
                ckpt = tf.train.get_checkpoint_state(checkpoint_dir=FLAGS.whole_pretraining)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    print('Session restored from trained model at {}!'.format(ckpt.model_checkpoint_path))
                else:
                    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), FLAGS.whole_pretraining)
                model_restarting_from_pretrained()
            else:
                varlist = tf.trainable_variables() + bn_moving_vars
                saver = tf.train.Saver(varlist)
                ckpt = tf.train.get_checkpoint_state(checkpoint_dir=FLAGS.checkpoint_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    print('Session restored from trained model at {}!'.format(ckpt.model_checkpoint_path))
                else:
                    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), FLAGS.checkpoint_dir)

            varlist = tf.trainable_variables() + bn_moving_vars
            saver = tf.train.Saver(var_list=varlist, max_to_keep=5)

            if not os.path.exists(FLAGS.summary_dir):
                os.makedirs(FLAGS.summary_dir)
            loss_summary_file = open(FLAGS.summary_dir+'loss_summary.txt', 'w')
            train_summary_file = open(FLAGS.summary_dir+'train_summary.txt', 'w')
            val_summary_file = open(FLAGS.summary_dir+'val_summary.txt', 'w')
            model_sampling_summary_file = open(FLAGS.summary_dir+'model_summary.txt', 'w')

            for step in xrange(FLAGS.max_steps):

                if FLAGS.use_restarting and step % FLAGS.restart_step == 0:
                    # budget_varlist = [v for v in tf.trainable_variables() if any(x in v.name for x in ["BudgetModule"])]
                    # init_budget_op = tf.variables_initializer(budget_varlist)
                    # sess.run(init_budget_op)
                    model_restarting_from_pretrained()
                    for _ in itertools.repeat(None, FLAGS.retraining_step):
                        # saver = tf.train.Saver(tf.trainable_variables() + bn_moving_vars)
                        start_time = time.time()
                        acc_budget_lst, loss_degrad_lst, loss_utility_lst, loss_budget_lst, psnr_lst = [], [], [], [], []
                        sess.run([zero_ops_budget])
                        for _ in itertools.repeat(None, FLAGS.n_minibatches):
                            images, labels_yaw, labels_pitch, labels_roll, labels_yaw_cont, \
                            labels_pitch_cont, labels_roll_cont, genders = sess.run([tr_images_op,
                                                                                                 tr_labels_yaw_op,
                                                                                                 tr_labels_pitch_op,
                                                                                                 tr_labels_roll_op,
                                                                                                 tr_labels_yaw_cont_op,
                                                                                                 tr_labels_pitch_cont_op,
                                                                                                 tr_labels_roll_cont_op,
                                                                                                 tr_genders_op])

                            _, psnr, acc_budget, loss_degrad_value, loss_utility_value, loss_budget_value = sess.run(
                                [accum_ops_budget, psnr_op, acc_budget_ensemble_op, loss_degrad_op,
                                 loss_utility_op, loss_budget_op],
                                feed_dict={images_placeholder: images,
                                           label_yaw_placeholder: labels_yaw,
                                           label_pitch_placeholder: labels_pitch,
                                           label_roll_placeholder: labels_roll,
                                           label_yaw_cont_placeholder: labels_yaw_cont,
                                           label_pitch_cont_placeholder: labels_pitch_cont,
                                           label_roll_cont_placeholder: labels_roll_cont,
                                           budget_labels_placeholder: genders,
                                           istraining_placeholder: True})
                            acc_budget_lst.append(acc_budget)
                            psnr_lst.append(psnr)
                            loss_degrad_lst.append(loss_degrad_value)
                            loss_utility_lst.append(loss_utility_value)
                            loss_budget_lst.append(loss_budget_value)
                        sess.run(apply_gradient_op_budget)
                        assert not np.isnan(np.mean(loss_degrad_lst)), 'Model diverged with loss = NaN'
                        loss_summary = 'Restarting (Budget), Step: {:4d}, time: {:.4f}, budget loss: {:.8f}, training budget accuracy: {:.5f}, utility loss: {:.8f}, psnr: {:.8f}'.format(step,
                                        time.time() - start_time, np.mean(loss_budget_lst), np.mean(acc_budget_lst), np.mean(loss_utility_lst), np.mean(psnr_lst))
                        model_sampling_summary_file.write(loss_summary + '\n')
                        print(loss_summary)

                start_time = time.time()
                loss_degrad_lst, loss_utility_lst, loss_budget_lst, psnr_lst = [], [], [], []
                sess.run(zero_ops_degradation)
                for _ in itertools.repeat(None, FLAGS.n_minibatches):
                    images, labels_yaw, labels_pitch, labels_roll, labels_yaw_cont, \
                    labels_pitch_cont, labels_roll_cont, genders = sess.run([tr_images_op,
                                                                    tr_labels_yaw_op, tr_labels_pitch_op,tr_labels_roll_op,
                                                                    tr_labels_yaw_cont_op, tr_labels_pitch_cont_op, tr_labels_roll_cont_op,
                                                                    tr_genders_op])

                    _, psnr, argmax_nentropy, loss_degrad_value, loss_utility_value, loss_budget_value = sess.run([accum_ops_degradation, psnr_op,
                                                    argmax_nentropy_op, loss_degrad_op, loss_utility_op, loss_budget_op],
                                                    feed_dict={images_placeholder: images,
                                                               label_yaw_placeholder: labels_yaw,
                                                               label_pitch_placeholder: labels_pitch,
                                                               label_roll_placeholder: labels_roll,
                                                               label_yaw_cont_placeholder: labels_yaw_cont,
                                                               label_pitch_cont_placeholder: labels_pitch_cont,
                                                               label_roll_cont_placeholder: labels_roll_cont,
                                                               budget_labels_placeholder: genders,
                                                               istraining_placeholder: True,
                                                               })
                    print(argmax_nentropy)
                    #print(loss_budget_uniform_tensor)
                    psnr_lst.append(psnr)
                    loss_degrad_lst.append(loss_degrad_value)
                    loss_utility_lst.append(loss_utility_value)
                    loss_budget_lst.append(loss_budget_value)
                _, _lambda = sess.run([apply_gradient_op_degradation, _lambda_op])

                assert not np.isnan(np.mean(loss_degrad_lst)), 'Model diverged with loss = NaN'
                duration = time.time() - start_time


                loss_summary = 'Alternating Training (Degradation), Lambda: {:.8f}, Step: {:4d}, time: {:.4f}, degradation loss: {:.8f}, utility loss: {:.8f}, budget loss: {:.8f}, psnr: {:.8f}'.format(_lambda, step,
                                                        duration, np.mean(loss_degrad_lst), np.mean(loss_utility_lst), np.mean(loss_budget_lst), np.mean(psnr_lst))

                print(loss_summary)
                loss_summary_file.write(loss_summary + '\n')

                if FLAGS.use_monitor_utility:
                    while True:
                    #for _ in itertools.repeat(None, FLAGS.adaptation_utility_steps):
                        start_time = time.time()
                        acc_budget_lst, loss_degrad_lst, loss_utility_lst, loss_budget_lst, psnr_lst = [], [], [], [], []
                        for _ in itertools.repeat(None, FLAGS.n_minibatches):
                            images, labels_yaw, labels_pitch, labels_roll, labels_yaw_cont, \
                            labels_pitch_cont, labels_roll_cont, genders = sess.run([val_images_op,
                                                                                                 val_labels_yaw_op,
                                                                                                 val_labels_pitch_op,
                                                                                                 val_labels_roll_op,
                                                                                                 val_labels_yaw_cont_op,
                                                                                                 val_labels_pitch_cont_op,
                                                                                                 val_labels_roll_cont_op,
                                                                                                 val_genders_op])

                            acc_budget, psnr, loss_degrad, loss_utility, loss_budget = sess.run(
                                                        [acc_budget_ensemble_op, psnr_op, loss_degrad_op, loss_utility_op, loss_budget_op],
                                                        feed_dict={images_placeholder: images,
                                                                   label_yaw_placeholder: labels_yaw,
                                                                   label_pitch_placeholder: labels_pitch,
                                                                   label_roll_placeholder: labels_roll,
                                                                   label_yaw_cont_placeholder: labels_yaw_cont,
                                                                   label_pitch_cont_placeholder: labels_pitch_cont,
                                                                   label_roll_cont_placeholder: labels_roll_cont,
                                                                   budget_labels_placeholder: genders,
                                                                   istraining_placeholder: True,
                                                        })
                            acc_budget_lst.append(acc_budget)
                            psnr_lst.append(psnr)
                            loss_degrad_lst.append(loss_degrad)
                            loss_utility_lst.append(loss_utility)
                            loss_budget_lst.append(loss_budget)
                            # test_writer.add_summary(summary, step)
                        val_summary = "Step: {:4d}, time: {:.4f}, degradation loss: {:.8f}, utility loss: {:.8f}, budget loss: {:.8f}, validation budget accuracy: {:.5f}, psnr: {:.8f}" .format(
                                step,
                                time.time() - start_time, np.mean(loss_degrad_lst),
                                np.mean(loss_utility_lst), np.mean(loss_budget_lst),
                                np.mean(acc_budget_lst), np.mean(psnr_lst))
                        print(val_summary)

                        if np.mean(loss_utility_lst) <= FLAGS.util_loss_val_thresh:
                            break
                        start_time = time.time()
                        sess.run(zero_ops_utility)
                        acc_budget_lst, loss_degrad_lst, loss_utility_lst, loss_budget_lst, psnr_lst = [], [], [], [], []
                        for _ in itertools.repeat(None, FLAGS.n_minibatches):
                            images, labels_yaw, labels_pitch, labels_roll, labels_yaw_cont, \
                            labels_pitch_cont, labels_roll_cont, genders = sess.run([val_images_op,
                                                                                                 val_labels_yaw_op,
                                                                                                 val_labels_pitch_op,
                                                                                                 val_labels_roll_op,
                                                                                                 val_labels_yaw_cont_op,
                                                                                                 val_labels_pitch_cont_op,
                                                                                                 val_labels_roll_cont_op,
                                                                                                 val_genders_op])

                            _, psnr, acc_budget, loss_degrad, loss_utility, loss_budget = sess.run([accum_ops_utility, psnr_op,
                                                            acc_budget_ensemble_op, loss_degrad_op, loss_utility_op, loss_budget_op],
                                                            feed_dict={images_placeholder: images,
                                                                       label_yaw_placeholder: labels_yaw,
                                                                       label_pitch_placeholder: labels_pitch,
                                                                       label_roll_placeholder: labels_roll,
                                                                       label_yaw_cont_placeholder: labels_yaw_cont,
                                                                       label_pitch_cont_placeholder: labels_pitch_cont,
                                                                       label_roll_cont_placeholder: labels_roll_cont,
                                                                       budget_labels_placeholder: genders,
                                                                       istraining_placeholder: True,
                                                               })
                            acc_budget_lst.append(acc_budget)
                            psnr_lst.append(psnr)
                            loss_degrad_lst.append(loss_degrad)
                            loss_utility_lst.append(loss_utility)
                            loss_budget_lst.append(loss_budget)
                        sess.run([apply_gradient_op_utility])
                        assert not np.isnan(np.mean(loss_degrad_lst)), 'Model diverged with loss = NaN'
                        loss_summary = 'Alternating Training (Utility), Step: {:4d}, time: {:.4f}, degradation loss: {:.8f}, utility loss: {:.8f}, budget loss: {:.8f}, psnr: {:.8f}'.format(step,
                                                            time.time() - start_time, np.mean(loss_degrad_lst),
                                                            np.mean(loss_utility_lst), np.mean(loss_budget_lst), np.mean(psnr_lst))

                        print(loss_summary)

                if FLAGS.use_monitor_budget:
                    while True:
                        start_time = time.time()
                        sess.run(zero_ops_budget)
                        acc_budget_lst, loss_degrad_lst, loss_utility_lst, loss_budget_lst, psnr_lst = [], [], [], [], []
                        for _ in itertools.repeat(None, FLAGS.n_minibatches):
                            images, labels_yaw, labels_pitch, labels_roll, labels_yaw_cont, \
                            labels_pitch_cont, labels_roll_cont, genders = sess.run([tr_images_op,
                                                                                                 tr_labels_yaw_op,
                                                                                                 tr_labels_pitch_op,
                                                                                                 tr_labels_roll_op,
                                                                                                 tr_labels_yaw_cont_op,
                                                                                                 tr_labels_pitch_cont_op,
                                                                                                 tr_labels_roll_cont_op,
                                                                                                 tr_genders_op])

                            _, psnr, acc_budget, loss_degrad, loss_utility, loss_budget = sess.run([accum_ops_budget, psnr_op, acc_budget_op, loss_degrad_op, loss_utility_op, loss_budget_op],
                                                    feed_dict={images_placeholder: images,
                                                                   label_yaw_placeholder: labels_yaw,
                                                                   label_pitch_placeholder: labels_pitch,
                                                                   label_roll_placeholder: labels_roll,
                                                                   label_yaw_cont_placeholder: labels_yaw_cont,
                                                                   label_pitch_cont_placeholder: labels_pitch_cont,
                                                                   label_roll_cont_placeholder: labels_roll_cont,
                                                                   budget_labels_placeholder: genders,
                                                                   istraining_placeholder: True,
                                                               })
                            acc_budget_lst.append(acc_budget)
                            psnr_lst.append(psnr)
                            loss_degrad_lst.append(loss_degrad)
                            loss_utility_lst.append(loss_utility)
                            loss_budget_lst.append(loss_budget)
                        sess.run([apply_gradient_op_budget])
                        assert not np.isnan(np.mean(loss_degrad_lst)), 'Model diverged with loss = NaN'
                        loss_summary = 'Alternating Training (Budget), Step: {:4d}, time: {:.4f}, degradation loss: {:.8f}, utility loss: {:.8f}, budget loss: {:.8f}, ' \
                                   ' budget accuracy: {:.5f}, psnr: {:.8f}'.format(step,
                                                            time.time() - start_time, np.mean(loss_degrad_lst),
                                                            np.mean(loss_utility_lst), np.mean(loss_budget_lst),
                                                            np.mean(acc_budget_lst), np.mean(psnr_lst))

                        print(loss_summary)
                        if np.mean(acc_budget_lst) >= FLAGS.budget_acc_train_thresh:
                            break

                if step % FLAGS.val_step == 0:
                    start_time = time.time()
                    acc_budget_lst, loss_degrad_lst, loss_utility_lst, loss_budget_lst, psnr_lst = [], [], [], [], []
                    for _ in itertools.repeat(None, 20):
                        images, labels_yaw, labels_pitch, labels_roll, labels_yaw_cont, \
                        labels_pitch_cont, labels_roll_cont, genders = sess.run([tr_images_op,
                                                                                             tr_labels_yaw_op,
                                                                                             tr_labels_pitch_op,
                                                                                             tr_labels_roll_op,
                                                                                             tr_labels_yaw_cont_op,
                                                                                             tr_labels_pitch_cont_op,
                                                                                             tr_labels_roll_cont_op,
                                                                                             tr_genders_op])

                        acc_budget, psnr, loss_degrad, loss_utility, loss_budget = sess.run(
                                                        [acc_budget_ensemble_op, psnr_op, loss_degrad_op, loss_utility_op, loss_budget_op],
                                                        feed_dict={images_placeholder: images,
                                                                   label_yaw_placeholder: labels_yaw,
                                                                   label_pitch_placeholder: labels_pitch,
                                                                   label_roll_placeholder: labels_roll,
                                                                   label_yaw_cont_placeholder: labels_yaw_cont,
                                                                   label_pitch_cont_placeholder: labels_pitch_cont,
                                                                   label_roll_cont_placeholder: labels_roll_cont,
                                                                   budget_labels_placeholder: genders,
                                                                   istraining_placeholder: True,
                                                                   })
                        acc_budget_lst.append(acc_budget)
                        psnr_lst.append(psnr)
                        loss_degrad_lst.append(loss_degrad)
                        loss_utility_lst.append(loss_utility)
                        loss_budget_lst.append(loss_budget)

                    train_summary = "Step: {:4d}, time: {:.4f}, degradation loss: {:.8f}, training utility loss: {:.8f}, training budget loss: {:.8f}, training budget accuracy: {:.5f}, psnr: {:.8f}".format(
                                    step, time.time() - start_time, np.mean(loss_degrad_lst),
                                    np.mean(loss_utility_lst), np.mean(loss_budget_lst), np.mean(acc_budget_lst), np.mean(psnr_lst))
                    print(train_summary)
                    train_summary_file.write(train_summary + '\n')

                    start_time = time.time()
                    acc_budget_lst, loss_degrad_lst, loss_utility_lst, loss_budget_lst, psnr_lst = [], [], [], [], []
                    for _ in itertools.repeat(None, 20):
                        images, labels_yaw, labels_pitch, labels_roll, labels_yaw_cont, \
                        labels_pitch_cont, labels_roll_cont, genders = sess.run([val_images_op,
                                                                                             val_labels_yaw_op,
                                                                                             val_labels_pitch_op,
                                                                                             val_labels_roll_op,
                                                                                             val_labels_yaw_cont_op,
                                                                                             val_labels_pitch_cont_op,
                                                                                             val_labels_roll_cont_op,
                                                                                             val_genders_op])

                        acc_budget, psnr, loss_degrad, loss_utility, loss_budget = sess.run(
                                                        [acc_budget_ensemble_op, psnr_op, loss_degrad_op, loss_utility_op, loss_budget_op],
                                                        feed_dict={images_placeholder: images,
                                                                   label_yaw_placeholder: labels_yaw,
                                                                   label_pitch_placeholder: labels_pitch,
                                                                   label_roll_placeholder: labels_roll,
                                                                   label_yaw_cont_placeholder: labels_yaw_cont,
                                                                   label_pitch_cont_placeholder: labels_pitch_cont,
                                                                   label_roll_cont_placeholder: labels_roll_cont,
                                                                   budget_labels_placeholder: genders,
                                                                   istraining_placeholder: True,
                                                                   })
                        acc_budget_lst.append(acc_budget)
                        psnr_lst.append(psnr)
                        loss_degrad_lst.append(loss_degrad)
                        loss_utility_lst.append(loss_utility)
                        loss_budget_lst.append(loss_budget)
                        # test_writer.add_summary(summary, step)
                    val_summary = "Step: {:4d}, time: {:.4f}, degradation loss: {:.8f}, validation utility loss: {:.8f}, validation budget loss: {:.8f}, validation budget accuracy: {:.5f}, psnr: {:.8f}".format(
                                    step, time.time() - start_time, np.mean(loss_degrad_lst),
                                    np.mean(loss_utility_lst), np.mean(loss_budget_lst), np.mean(acc_budget_lst), np.mean(psnr_lst))
                    print(val_summary)
                    val_summary_file.write(val_summary + '\n')

                if step % FLAGS.save_step == 0 or (step + 1) == FLAGS.max_steps:
                    checkpoint_path = os.path.join(FLAGS.checkpoint_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)

            loss_summary_file.close()
            train_summary_file.close()
            val_summary_file.close()
            coord.request_stop()
            coord.join(threads)

    print("done")

def run_validation_multi_model_mixed_restarting():

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True

    multiplier_lst = [0.5 - i * 0.02 for i in range(FLAGS.NBudget)]
    from collections import defaultdict
    logits_budget_lst_dct = defaultdict(list)
    wplaceholder_dct = {}

    with tf.Graph().as_default():
        with tf.Session(config=config) as sess:
            global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
            images_placeholder, label_yaw_placeholder, label_pitch_placeholder, label_roll_placeholder, label_yaw_cont_placeholder, label_pitch_cont_placeholder, label_roll_cont_placeholder = placeholder_inputs(
                FLAGS.batch_size * FLAGS.gpu_num)
            budget_labels_placeholder =  tf.placeholder(tf.int64, shape=(FLAGS.batch_size * FLAGS.gpu_num))

            istraining_placeholder = tf.placeholder(tf.bool)
            #budgetNet_model_name_lst = ['mobilenet_v1_050', 'resnet_v2_50', 'mobilenet_v1', 'mobilenet_v1_075']
            for model_name in multiplier_lst:
                wplaceholder_dct['{}'.format(model_name)] = tf.placeholder(tf.float32)

            # Compute Acc
            logits_budget_lst = []


            if FLAGS.use_lambda_decay:
                _lambda_op = tf.train.exponential_decay(FLAGS._lambda, global_step=global_step, decay_steps=10, decay_rate=0.9)
            else:
                _lambda_op = tf.identity(FLAGS._lambda)
            with tf.variable_scope(tf.get_variable_scope()) as scope:
                for gpu_index in range(0, FLAGS.gpu_num):
                    with tf.device('/gpu:%d' % gpu_index):
                        print('/gpu:%d' % gpu_index)
                        with tf.name_scope('%s_%d' % ('gpu', gpu_index)) as scope:
                            images_degraded = residualnet(images_placeholder[gpu_index * FLAGS.batch_size:(gpu_index + 1) * FLAGS.batch_size])


                            if FLAGS.use_avg_replicate:
                                images_degraded = _avg_replicate(images_degraded)

                            fc_yaw, fc_pitch, fc_roll = hopenet(images_degraded, is_training=istraining_placeholder)

                            idx_tensor = tf.expand_dims(tf.range(66, dtype=tf.float32), 0)
                            idx_tensor = tf.tile(idx_tensor, [FLAGS.batch_size, 1])

                            yaw_predicted = tf.reduce_sum(tf.multiply(tf.nn.softmax(fc_yaw, axis=1), idx_tensor), axis=1) * 3 - 99
                            pitch_predicted = tf.reduce_sum(tf.multiply(tf.nn.softmax(fc_pitch, axis=1), idx_tensor), axis=1) * 3 - 99
                            roll_predicted = tf.reduce_sum(tf.multiply(tf.nn.softmax(fc_roll, axis=1), idx_tensor), axis=1) * 3 - 99


                            logits_budget = tf.zeros([FLAGS.batch_size, FLAGS.num_classes_budget])
                            logits_lst = []
                            for multiplier in multiplier_lst:
                                print(multiplier)
                                print(tf.trainable_variables())
                                logits = mobilenet(images_degraded, istraining_placeholder, depth_multiplier=multiplier)
                                logits_lst.append(logits)
                                logits_budget_lst_dct['{}'.format(multiplier)].append(logits)
                                logits_budget += logits / len(multiplier_lst)

                            logits_budget_lst.append(logits_budget)

                            # Reuse variables for the next tower.
                            tf.get_variable_scope().reuse_variables()


            logits_budget = tf.concat(logits_budget_lst, 0)
            acc_budget_ensemble_op = accuracy(logits_budget, budget_labels_placeholder)

            acc_budget_op_lst = []
            for multiplier in multiplier_lst:
                acc_budget_op = accuracy(tf.concat(logits_budget_lst_dct['{}'.format(multiplier)], 0), budget_labels_placeholder)
                acc_budget_op_lst.append(acc_budget_op)

            val_files = [os.path.join(FLAGS.val_files_dir, f) for f in
                         os.listdir(FLAGS.val_files_dir) if f.endswith('.tfrecords')]

            print(val_files)

            val_images_op, val_labels_yaw_op, val_labels_pitch_op, val_labels_roll_op, \
            val_labels_yaw_cont_op, val_labels_pitch_cont_op, val_labels_roll_cont_op, val_genders_op = input_data.inputs_images(filenames=val_files,
                                                                  batch_size=FLAGS.batch_size * FLAGS.gpu_num,
                                                                  num_epochs=1,
                                                                  num_threads=FLAGS.num_threads,
                                                                  num_examples_per_epoch=FLAGS.num_examples_per_epoch)

            init_op = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())
            sess.run(init_op)
            # Create a saver for writing training checkpoints.

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            gvar_list = tf.global_variables()
            bn_moving_vars = [g for g in gvar_list if 'moving_mean' in g.name]
            bn_moving_vars += [g for g in gvar_list if 'moving_variance' in g.name]

            varlist = tf.trainable_variables() + bn_moving_vars
            #varlist = tf.trainable_variables()
            saver = tf.train.Saver(varlist)
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir=FLAGS.checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Session restored from trained model at {}!'.format(ckpt.model_checkpoint_path))
            else:
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), FLAGS.checkpoint_dir)

            acc_budget_lst = []
            try:
                while not coord.should_stop():
                    images, labels_yaw, labels_pitch, labels_roll, \
                    labels_yaw_cont, labels_pitch_cont, labels_roll_cont, genders = sess.run([val_images_op,
                                                                         val_labels_yaw_op,
                                                                         val_labels_pitch_op,
                                                                         val_labels_roll_op,
                                                                         val_labels_yaw_cont_op,
                                                                         val_labels_pitch_cont_op,
                                                                         val_labels_roll_cont_op,
                                                                         val_genders_op])

                    acc_budget = sess.run([acc_budget_ensemble_op],
                                                        feed_dict={images_placeholder: images,
                                                                   label_yaw_placeholder: labels_yaw,
                                                                   label_pitch_placeholder: labels_pitch,
                                                                   label_roll_placeholder: labels_roll,
                                                                   label_yaw_cont_placeholder: labels_yaw_cont,
                                                                   label_pitch_cont_placeholder: labels_pitch_cont,
                                                                   label_roll_cont_placeholder: labels_roll_cont,
                                                                   budget_labels_placeholder: genders,
                                                                   istraining_placeholder: True,
                                                                   })
                    print(acc_budget)
                    acc_budget_lst.append(acc_budget)
            except tf.errors.OutOfRangeError:
                print('Done testing on all the examples')
            finally:
                coord.request_stop()
                coord.join(threads)



    print("done")

def run_gpu_utility_train_tfrecords():
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True
    use_pretrained_model = True

    tf.reset_default_graph()
    with tf.Graph().as_default():
        with tf.Session(config=config) as sess:
            global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
            istraining_placeholder = tf.placeholder(tf.bool)
            images_placeholder, label_yaw_placeholder, label_pitch_placeholder, label_roll_placeholder, \
            label_yaw_cont_placeholder, label_pitch_cont_placeholder, label_roll_cont_placeholder = placeholder_inputs(
                FLAGS.batch_size * FLAGS.gpu_num)
            losses = []
            losses_yaw = []
            losses_pitch = []
            losses_roll = []
            tower_grads_low = []
            tower_grads_mid = []
            tower_grads_high = []
            opt_low = tf.train.AdamOptimizer(1e-5)
            opt_mid = tf.train.AdamOptimizer(1e-5)
            opt_high = tf.train.AdamOptimizer(5e-5)
            with tf.variable_scope(tf.get_variable_scope()) as scope:
                for gpu_index in range(0, FLAGS.gpu_num):
                    with tf.device('/gpu:%d' % gpu_index):
                        print('/gpu:%d' % gpu_index)
                        with tf.name_scope('%s_%d' % ('gpu', gpu_index)) as scope:
                            X = images_placeholder[gpu_index * FLAGS.batch_size:(gpu_index + 1) * FLAGS.batch_size]
                            # X = tf.image.resize_bilinear(images_placeholder[gpu_index * FLAGS.batch_size:(gpu_index + 1) * FLAGS.batch_size], [224, 224])
                            # X = tf.subtract(X, tf.stack([tf.scalar_mul(0.485, tf.ones([1, 224, 224], tf.float32)),
                            #                              tf.scalar_mul(0.456, tf.ones([1, 224, 224], tf.float32)),
                            #                              tf.scalar_mul(0.406, tf.ones([1, 224, 224], tf.float32))],
                            #                             axis=3))
                            #
                            # X = tf.divide(X, tf.stack([tf.scalar_mul(0.229, tf.ones([1, 224, 224], tf.float32)),
                            #                            tf.scalar_mul(0.224, tf.ones([1, 224, 224], tf.float32)),
                            #                            tf.scalar_mul(0.225, tf.ones([1, 224, 224], tf.float32))],
                            #                           axis=3))
                            fc_yaw, fc_pitch, fc_roll = hopenet(X, is_training=istraining_placeholder)

                            loss_yaw = tower_loss_xentropy_sparse(fc_yaw, label_yaw_placeholder[
                                                                          gpu_index * FLAGS.batch_size:(
                                                                                                       gpu_index + 1) * FLAGS.batch_size])
                            loss_pitch = tower_loss_xentropy_sparse(fc_pitch, label_pitch_placeholder[
                                                                              gpu_index * FLAGS.batch_size:(
                                                                                                           gpu_index + 1) * FLAGS.batch_size])
                            loss_roll = tower_loss_xentropy_sparse(fc_roll, label_roll_placeholder[
                                                                            gpu_index * FLAGS.batch_size:(
                                                                                                         gpu_index + 1) * FLAGS.batch_size])

                            idx_tensor = tf.expand_dims(tf.range(66, dtype=tf.float32), 0)
                            idx_tensor = tf.tile(idx_tensor, [FLAGS.batch_size, 1])

                            yaw_predicted = tf.reduce_sum(tf.multiply(tf.nn.softmax(fc_yaw, axis=1), idx_tensor),
                                                          axis=1) * 3 - 99
                            pitch_predicted = tf.reduce_sum(tf.multiply(tf.nn.softmax(fc_pitch, axis=1), idx_tensor),
                                                            axis=1) * 3 - 99
                            roll_predicted = tf.reduce_sum(tf.multiply(tf.nn.softmax(fc_roll, axis=1), idx_tensor),
                                                           axis=1) * 3 - 99

                            loss_reg_yaw = tower_loss_mse(yaw_predicted, label_yaw_cont_placeholder[
                                                                         gpu_index * FLAGS.batch_size:(
                                                                                                      gpu_index + 1) * FLAGS.batch_size])
                            loss_reg_pitch = tower_loss_mse(pitch_predicted, label_pitch_cont_placeholder[
                                                                             gpu_index * FLAGS.batch_size:(
                                                                                                          gpu_index + 1) * FLAGS.batch_size])
                            loss_reg_roll = tower_loss_mse(roll_predicted, label_roll_cont_placeholder[
                                                                           gpu_index * FLAGS.batch_size:(
                                                                                                        gpu_index + 1) * FLAGS.batch_size])

                            loss_yaw += FLAGS.alpha * loss_reg_yaw
                            loss_pitch += FLAGS.alpha * loss_reg_pitch
                            loss_roll += FLAGS.alpha * loss_reg_roll

                            loss = loss_yaw + loss_pitch + loss_roll
                            losses.append(loss)
                            losses_yaw.append(loss_yaw)
                            losses_pitch.append(loss_pitch)
                            losses_roll.append(loss_roll)

                            varlist_low = [v for v in tf.trainable_variables() if
                                           any(x in v.name for x in ["UtilityModule/conv1/"])]
                            varlist_mid = [v for v in tf.trainable_variables() if
                                           any(x in v.name for x in ["block1", "block2", "block3", "block4"])]
                            varlist_high = [v for v in tf.trainable_variables() if
                                            any(x in v.name for x in ["fc_yaw", "fc_pitch", "fc_roll"])]

                            print([v.name for v in varlist_low])
                            print([v.name for v in varlist_mid])
                            print([v.name for v in varlist_high])
                            grads_low = opt_low.compute_gradients(loss, varlist_low)
                            grads_mid = opt_mid.compute_gradients(loss, varlist_mid)
                            grads_high = opt_high.compute_gradients(loss, varlist_high)

                            tower_grads_low.append(grads_low)
                            tower_grads_mid.append(grads_mid)
                            tower_grads_high.append(grads_high)
                            tf.get_variable_scope().reuse_variables()

            loss_op = tf.reduce_mean(losses, name='loss')
            loss_yaw_op = tf.reduce_mean(losses_yaw, name='loss_yaw')
            loss_pitch_op = tf.reduce_mean(losses_pitch, name='loss_pitch')
            loss_roll_op = tf.reduce_mean(losses_roll, name='loss_roll')
            grads_low = average_gradients(tower_grads_low)
            grads_mid = average_gradients(tower_grads_mid)
            grads_high = average_gradients(tower_grads_high)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies([tf.group(*update_ops)]):
                train_op = tf.group(opt_low.apply_gradients(grads_low, global_step=global_step),
                                    opt_mid.apply_gradients(grads_mid, global_step=global_step),
                                    opt_high.apply_gradients(grads_high, global_step=global_step))

            train_files = [os.path.join(FLAGS.train_files_dir, f) for f in
                           os.listdir(FLAGS.train_files_dir) if f.endswith('.tfrecords')]
            val_files = [os.path.join(FLAGS.val_files_dir, f) for f in
                           os.listdir(FLAGS.val_files_dir) if f.endswith('.tfrecords')]

            tr_images_op, tr_labels_yaw_op, tr_labels_pitch_op, tr_labels_roll_op, \
            tr_labels_yaw_cont_op, tr_labels_pitch_cont_op, tr_labels_roll_cont_op, _,  = input_data.inputs_images(filenames=train_files,
                                                                  batch_size=FLAGS.batch_size * FLAGS.gpu_num,
                                                                  num_epochs=None,
                                                                  num_threads=FLAGS.num_threads,
                                                                  num_examples_per_epoch=FLAGS.num_examples_per_epoch)

            val_images_op, val_labels_yaw_op, val_labels_pitch_op, val_labels_roll_op, \
            val_labels_yaw_cont_op, val_labels_pitch_cont_op, val_labels_roll_cont_op, _,  = input_data.inputs_images(filenames=val_files,
                                                                  batch_size=FLAGS.batch_size * FLAGS.gpu_num,
                                                                  num_epochs=None,
                                                                  num_threads=FLAGS.num_threads,
                                                                  num_examples_per_epoch=FLAGS.num_examples_per_epoch)

            init_op = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())
            sess.run(init_op)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            gvar_list = tf.global_variables()
            bn_moving_vars = [g for g in gvar_list if 'moving_mean' in g.name]
            bn_moving_vars += [g for g in gvar_list if 'moving_variance' in g.name]

            import pickle
            def load_obj(name):
                with open(name + '.pkl', 'rb') as f:
                    u = pickle._Unpickler(f)
                    u.encoding = 'latin1'
                    return u.load()
            '''
            var_dict = load_obj('hopenet')
            for v in tf.trainable_variables() + bn_moving_vars:
                # if any(x in v.name for x in ['weights', 'bias', 'beta', 'gamma', 'moving_mean', 'moving_variance']):
                print("v.name: {}".format(v.name))
                print("v.shape: {}".format(sess.run(v).shape))
                sess.run(v.assign(var_dict[v.name[:-2]]))

            print(len(list(var_dict.keys())))
            print(len(tf.trainable_variables() + bn_moving_vars))


            saver = tf.train.Saver(var_list=tf.trainable_variables() + bn_moving_vars)
            checkpoint_path = os.path.join(FLAGS.utility_models, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=global_step)

            '''

            if use_pretrained_model:
                saver = tf.train.Saver(var_list=tf.trainable_variables() + bn_moving_vars)
                ckpt = tf.train.get_checkpoint_state(checkpoint_dir=FLAGS.pretrained_hopenet)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    print('Session restored from trained model at {}!'.format(ckpt.model_checkpoint_path))
                else:
                    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), FLAGS.pretrained_hopenet)
            else:
                saver = tf.train.Saver(var_list=tf.trainable_variables() + bn_moving_vars)
                ckpt = tf.train.get_checkpoint_state(checkpoint_dir=FLAGS.utility_models)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    print('Session restored from trained model at {}!'.format(ckpt.model_checkpoint_path))
                else:
                    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), FLAGS.utility_models)

            saver = tf.train.Saver(var_list=tf.trainable_variables() + bn_moving_vars)

            print('Ready to train network.')
            for step in range(FLAGS.max_steps):
                images, labels_yaw, labels_pitch, labels_roll, labels_yaw_cont, \
                labels_pitch_cont, labels_roll_cont = sess.run([tr_images_op,
                                                                tr_labels_yaw_op, tr_labels_pitch_op, tr_labels_roll_op,
                                                                tr_labels_yaw_cont_op, tr_labels_pitch_cont_op, tr_labels_roll_cont_op])
                _, loss_yaw, loss_pitch, loss_roll = sess.run([train_op, loss_yaw_op, loss_pitch_op, loss_roll_op],
                                                                  feed_dict={images_placeholder: images,
                                                                             label_yaw_placeholder: labels_yaw,
                                                                             label_pitch_placeholder: labels_pitch,
                                                                             label_roll_placeholder: labels_roll,
                                                                             label_yaw_cont_placeholder: labels_yaw_cont,
                                                                             label_pitch_cont_placeholder: labels_pitch_cont,
                                                                             label_roll_cont_placeholder: labels_roll_cont,
                                                                             istraining_placeholder: True})
                print('Losses: Yaw {:.4f}, Pitch {:.4f}, Roll {:.4f}'.format(loss_yaw, loss_pitch, loss_roll))

                if step % FLAGS.val_step == 0:
                    images, labels_yaw, labels_pitch, labels_roll, labels_yaw_cont, \
                    labels_pitch_cont, labels_roll_cont = sess.run([tr_images_op,
                                                                    tr_labels_yaw_op, tr_labels_pitch_op, tr_labels_roll_op,
                                                                    tr_labels_yaw_cont_op, tr_labels_pitch_cont_op, tr_labels_roll_cont_op])
                    loss_yaw, loss_pitch, loss_roll = sess.run([loss_yaw_op, loss_pitch_op, loss_roll_op],
                                                                  feed_dict={images_placeholder: images,
                                                                             label_yaw_placeholder: labels_yaw,
                                                                             label_pitch_placeholder: labels_pitch,
                                                                             label_roll_placeholder: labels_roll,
                                                                             label_yaw_cont_placeholder: labels_yaw_cont,
                                                                             label_pitch_cont_placeholder: labels_pitch_cont,
                                                                             label_roll_cont_placeholder: labels_roll_cont,
                                                                             istraining_placeholder: False})
                    print('Training Losses: Yaw {:.4f}, Pitch {:.4f}, Roll {:.4f}'.format(loss_yaw, loss_pitch, loss_roll))

                    images, labels_yaw, labels_pitch, labels_roll, labels_yaw_cont, \
                    labels_pitch_cont, labels_roll_cont = sess.run([val_images_op,
                                                                    val_labels_yaw_op, val_labels_pitch_op, val_labels_roll_op,
                                                                    val_labels_yaw_cont_op, val_labels_pitch_cont_op, val_labels_roll_cont_op])
                    loss_yaw, loss_pitch, loss_roll = sess.run([loss_yaw_op, loss_pitch_op, loss_roll_op],
                                                                  feed_dict={images_placeholder: images,
                                                                             label_yaw_placeholder: labels_yaw,
                                                                             label_pitch_placeholder: labels_pitch,
                                                                             label_roll_placeholder: labels_roll,
                                                                             label_yaw_cont_placeholder: labels_yaw_cont,
                                                                             label_pitch_cont_placeholder: labels_pitch_cont,
                                                                             label_roll_cont_placeholder: labels_roll_cont,
                                                                             istraining_placeholder: False})
                    print('Validation Losses: Yaw {:.4f}, Pitch {:.4f}, Roll {:.4f}'.format(loss_yaw, loss_pitch, loss_roll))

                if step % FLAGS.save_step == 0:
                    checkpoint_path = os.path.join(FLAGS.utility_models, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)

def run_gpu_utility_train():
    use_pretrained_model = True
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True

    transformations = transforms.Compose([transforms.Scale(240),
    transforms.RandomCrop(224), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    pose_dataset = datasets.BIWI('/home/wuzhenyu_sjtu/deep-head-pose/BIWI/hpdb', '/home/wuzhenyu_sjtu/deep-head-pose/BIWI/filename_list', transformations)

    tf.reset_default_graph()
    with tf.Graph().as_default():
        with tf.Session(config=config) as sess:
            global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
            istraining_placeholder = tf.placeholder(tf.bool)
            images_placeholder, label_yaw_placeholder, label_pitch_placeholder, label_roll_placeholder, \
            label_yaw_cont_placeholder, label_pitch_cont_placeholder, label_roll_cont_placeholder = placeholder_inputs(FLAGS.batch_size*FLAGS.gpu_num)
            losses = []
            losses_yaw = []
            losses_pitch = []
            losses_roll = []
            tower_grads_low = []
            tower_grads_mid = []
            tower_grads_high = []
            opt_low = tf.train.AdamOptimizer(1e-4)
            opt_mid = tf.train.AdamOptimizer(1e-4)
            opt_high = tf.train.AdamOptimizer(1e-4)
            with tf.variable_scope(tf.get_variable_scope()) as scope:
                for gpu_index in range(0, FLAGS.gpu_num):
                    with tf.device('/gpu:%d' % gpu_index):
                        print('/gpu:%d' % gpu_index)
                        with tf.name_scope('%s_%d' % ('gpu', gpu_index)) as scope:
                            X = images_placeholder[gpu_index * FLAGS.batch_size:(gpu_index + 1) * FLAGS.batch_size]
                            # X = tf.image.resize_bilinear(images_placeholder[gpu_index * FLAGS.batch_size:(gpu_index + 1) * FLAGS.batch_size], [224, 224])
                            # X = tf.subtract(X, tf.stack([tf.scalar_mul(0.485, tf.ones([1, 224, 224], tf.float32)),
                            #                              tf.scalar_mul(0.456, tf.ones([1, 224, 224], tf.float32)),
                            #                              tf.scalar_mul(0.406, tf.ones([1, 224, 224], tf.float32))],
                            #                             axis=3))
                            #
                            # X = tf.divide(X, tf.stack([tf.scalar_mul(0.229, tf.ones([1, 224, 224], tf.float32)),
                            #                            tf.scalar_mul(0.224, tf.ones([1, 224, 224], tf.float32)),
                            #                            tf.scalar_mul(0.225, tf.ones([1, 224, 224], tf.float32))],
                            #                           axis=3))
                            fc_yaw, fc_pitch, fc_roll = hopenet(X, is_training=istraining_placeholder)


                            loss_yaw = tower_loss_xentropy_sparse(fc_yaw, label_yaw_placeholder[gpu_index * FLAGS.batch_size:(gpu_index + 1) * FLAGS.batch_size])
                            loss_pitch = tower_loss_xentropy_sparse(fc_pitch, label_pitch_placeholder[gpu_index * FLAGS.batch_size:(gpu_index + 1) * FLAGS.batch_size])
                            loss_roll = tower_loss_xentropy_sparse(fc_roll, label_roll_placeholder[gpu_index * FLAGS.batch_size:(gpu_index + 1) * FLAGS.batch_size])


                            idx_tensor = tf.expand_dims(tf.range(66, dtype=tf.float32),0)
                            idx_tensor = tf.tile(idx_tensor, [FLAGS.batch_size, 1])

                            yaw_predicted = tf.reduce_sum(tf.multiply(tf.nn.softmax(fc_yaw, axis=1), idx_tensor), axis=1) * 3 - 99
                            pitch_predicted = tf.reduce_sum(tf.multiply(tf.nn.softmax(fc_pitch, axis=1), idx_tensor), axis=1) * 3 - 99
                            roll_predicted = tf.reduce_sum(tf.multiply(tf.nn.softmax(fc_roll, axis=1), idx_tensor), axis=1) * 3 - 99

                            loss_reg_yaw = tower_loss_mse(yaw_predicted, label_yaw_cont_placeholder[gpu_index * FLAGS.batch_size:(gpu_index + 1) * FLAGS.batch_size])
                            loss_reg_pitch = tower_loss_mse(pitch_predicted, label_pitch_cont_placeholder[gpu_index * FLAGS.batch_size:(gpu_index + 1) * FLAGS.batch_size])
                            loss_reg_roll = tower_loss_mse(roll_predicted, label_roll_cont_placeholder[gpu_index * FLAGS.batch_size:(gpu_index + 1) * FLAGS.batch_size])

                            loss_yaw += FLAGS.alpha * loss_reg_yaw
                            loss_pitch += FLAGS.alpha * loss_reg_pitch
                            loss_roll += FLAGS.alpha * loss_reg_roll

                            loss = loss_yaw + loss_pitch + loss_roll
                            losses.append(loss)
                            losses_yaw.append(loss_yaw)
                            losses_pitch.append(loss_pitch)
                            losses_roll.append(loss_roll)

                            varlist_low = [v for v in tf.trainable_variables() if any(x in v.name for x in ["resnet_v1_50/conv1/"])]
                            varlist_mid = [v for v in tf.trainable_variables() if any(x in v.name for x in ["block1", "block2", "block3", "block4"])]
                            varlist_high = [v for v in tf.trainable_variables() if any(x in v.name for x in ["fc_yaw", "fc_pitch", "fc_roll"])]

                            print([v.name for v in varlist_low])
                            print([v.name for v in varlist_mid])
                            print([v.name for v in varlist_high])
                            grads_low = opt_low.compute_gradients(loss, varlist_low)
                            grads_mid = opt_mid.compute_gradients(loss, varlist_mid)
                            grads_high = opt_high.compute_gradients(loss, varlist_high)

                            tower_grads_low.append(grads_low)
                            tower_grads_mid.append(grads_mid)
                            tower_grads_high.append(grads_high)
                            tf.get_variable_scope().reuse_variables()

            loss_op = tf.reduce_mean(losses, name='loss')
            loss_yaw_op = tf.reduce_mean(losses_yaw, name='loss_yaw')
            loss_pitch_op = tf.reduce_mean(losses_pitch, name='loss_pitch')
            loss_roll_op = tf.reduce_mean(losses_roll, name='loss_roll')
            grads_low = average_gradients(tower_grads_low)
            grads_mid = average_gradients(tower_grads_mid)
            grads_high = average_gradients(tower_grads_high)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies([tf.group(*update_ops)]):
                train_op = tf.group(opt_low.apply_gradients(grads_low, global_step=global_step),
                                opt_mid.apply_gradients(grads_mid, global_step=global_step),
                                opt_high.apply_gradients(grads_high, global_step=global_step))

            init_op = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())
            sess.run(init_op)

            gvar_list = tf.global_variables()
            bn_moving_vars = [g for g in gvar_list if 'moving_mean' in g.name]
            bn_moving_vars += [g for g in gvar_list if 'moving_variance' in g.name]

            '''
            for v in tf.trainable_variables() + bn_moving_vars:
                print("v.name: {}".format(v.name))
                print("v.shape: {}".format(sess.run(v).shape))
                print('=====================================')
            
            import pickle
            def load_obj(name):
                with open(name + '.pkl', 'rb') as f:
                    u = pickle._Unpickler(f)
                    u.encoding = 'latin1'
                    return u.load()

            var_dict = load_obj('hopenet')
            for v in tf.trainable_variables() + bn_moving_vars:
                # if any(x in v.name for x in ['weights', 'bias', 'beta', 'gamma', 'moving_mean', 'moving_variance']):
                print("v.name: {}".format(v.name))
                print("v.shape: {}".format(sess.run(v).shape))
                sess.run(v.assign(var_dict[v.name[:-2]]))

            print(len(list(var_dict.keys())))
            print(len(tf.trainable_variables() + bn_moving_vars))
            

            saver = tf.train.Saver(var_list=tf.trainable_variables() + bn_moving_vars)
            checkpoint_path = os.path.join(FLAGS.checkpoint_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=global_step)
            '''

            if use_pretrained_model:
                saver = tf.train.Saver(var_list=tf.trainable_variables() + bn_moving_vars)
                ckpt = tf.train.get_checkpoint_state(checkpoint_dir=FLAGS.utility_models)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    print('Session restored from trained model at {}!'.format(ckpt.model_checkpoint_path))
                else:
                    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), FLAGS.utility_models)
            else:
                saver = tf.train.Saver(var_list=tf.trainable_variables() + bn_moving_vars)
                ckpt = tf.train.get_checkpoint_state(checkpoint_dir=FLAGS.checkpoint_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    print('Session restored from trained model at {}!'.format(ckpt.model_checkpoint_path))
                else:
                    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), FLAGS.checkpoint_dir)

            train_loader = torch.utils.data.DataLoader(dataset=pose_dataset,
                                                       batch_size=FLAGS.batch_size * FLAGS.gpu_num,
                                                       shuffle=True,
                                                       drop_last=True,
                                                       num_workers=2)
            print('Ready to train network.')
            num_epochs = 10
            for epoch in range(num_epochs):
                for i, (images, labels, cont_labels, name) in enumerate(train_loader):
                    images = np.transpose(images, [0, 2, 3, 1])

                    # Binned labels
                    label_yaw =labels[:, 0]
                    label_pitch = labels[:, 1]
                    label_roll = labels[:, 2]

                    # Continuous labels
                    label_yaw_cont = cont_labels[:, 0]
                    label_pitch_cont = cont_labels[:, 1]
                    label_roll_cont = cont_labels[:, 2]
                    #print(label_yaw, label_pitch, label_roll, label_yaw_cont, label_pitch_cont, label_roll_cont)
                    _, loss_yaw, loss_pitch, loss_roll = sess.run([train_op, loss_yaw_op, loss_pitch_op, loss_roll_op],
                                                                  feed_dict={images_placeholder: images,
                                                                             label_yaw_placeholder: label_yaw,
                                                                             label_pitch_placeholder: label_pitch,
                                                                             label_roll_placeholder: label_roll,
                                                                             label_yaw_cont_placeholder: label_yaw_cont,
                                                                             label_pitch_cont_placeholder: label_pitch_cont,
                                                                             label_roll_cont_placeholder: label_roll_cont,
                                                                             istraining_placeholder: True})
                    if (i + 1) % 10 == 0:
                        print('Epoch [%d/%d], Iter [%d/%d] Losses: Yaw %.4f, Pitch %.4f, Roll %.4f'
                              % (epoch + 1, num_epochs, i + 1, len(pose_dataset) // FLAGS.batch_size, loss_yaw,
                                 loss_pitch, loss_roll))

def run_gpu_utility_eval():

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True

    with tf.Graph().as_default():
        with tf.Session(config=config) as sess:
            istraining_placeholder = tf.placeholder(tf.bool)
            network_fn = nets_factory.get_network_fn(
                FLAGS.model_name,
                num_classes=None,
                weight_decay=FLAGS.weight_decay,
                is_training=istraining_placeholder)
            images_placeholder = tf.placeholder(tf.float32, shape=(1, None, None, 3))
            with tf.device('/gpu:%d' % 0):
                print('/gpu:%d' % 0)
                with tf.name_scope('%s_%d' % ('gpu', 0)) as scope:
                    X = tf.image.resize_bilinear(images_placeholder, [224, 224])
                    X = tf.subtract(X, tf.stack([tf.scalar_mul(0.485, tf.ones([1, 224, 224], tf.float32)),
                                  tf.scalar_mul(0.456, tf.ones([1, 224, 224], tf.float32)),
                                  tf.scalar_mul(0.406, tf.ones([1, 224, 224], tf.float32))], axis=3))

                    X = tf.divide(X, tf.stack([tf.scalar_mul(0.229, tf.ones([1, 224, 224], tf.float32)),
                                  tf.scalar_mul(0.224, tf.ones([1, 224, 224], tf.float32)),
                                  tf.scalar_mul(0.225, tf.ones([1, 224, 224], tf.float32))], axis=3))

                    net, x_flatten = network_fn(X)
                    net = slim.avg_pool2d(net, 7)
                    net = tf.reshape(net, [1, -1])
                    fc_yaw = tf.nn.softmax(slim.fully_connected(net, 66, scope='resnet_v1_50/fc_yaw'))
                    fc_pitch = tf.nn.softmax(slim.fully_connected(net, 66, scope='resnet_v1_50/fc_pitch'))
                    fc_roll = tf.nn.softmax(slim.fully_connected(net, 66, scope='resnet_v1_50/fc_roll'))
                    yaw_predicted = tf.reduce_sum(tf.multiply(fc_yaw, tf.range(66, dtype=tf.float32))) * 3 - 99
                    pitch_predicted = tf.reduce_sum(tf.multiply(fc_pitch, tf.range(66, dtype=tf.float32))) * 3 - 99
                    roll_predicted = tf.reduce_sum(tf.multiply(fc_roll, tf.range(66, dtype=tf.float32))) * 3 - 99

            init_op = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())
            sess.run(init_op)

            gvar_list = tf.global_variables()
            bn_moving_vars = [g for g in gvar_list if 'moving_mean' in g.name]
            bn_moving_vars += [g for g in gvar_list if 'moving_variance' in g.name]

            for v in tf.trainable_variables()+bn_moving_vars:
                print("v.name: {}".format(v.name))
                print("v.shape: {}".format(sess.run(v).shape))
                print('=====================================')


            import pickle
            def load_obj(name):
                with open(name + '.pkl', 'rb') as f:
                    u = pickle._Unpickler(f)
                    u.encoding = 'latin1'
                    return u.load()

            var_dict = load_obj('hopenet')
            #print(var_dict)
            for v in tf.trainable_variables()+bn_moving_vars:
                #if any(x in v.name for x in ['weights', 'bias', 'beta', 'gamma', 'moving_mean', 'moving_variance']):
                print("v.name: {}".format(v.name))
                print("v.shape: {}".format(sess.run(v).shape))
                sess.run(v.assign(var_dict[v.name[:-2].replace('resnet_v1_50', 'UtilityModule')]))

            print(len(list(var_dict.keys())))
            print(len(tf.trainable_variables()+bn_moving_vars))
            video = cv2.VideoCapture("output/video/videoplayback.mp4")

            # New cv2
            width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))  # float
            height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float

            # Define the codec and create VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            out = cv2.VideoWriter('output/video/output-%s.avi' % "detect_face", fourcc, 1, (width, height))

            # # Old cv2
            # width = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))   # float
            # height = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)) # float
            #
            # # Define the codec and create VideoWriter object
            # fourcc = cv2.cv.CV_FOURCC(*'MJPG')
            # out = cv2.VideoWriter('output/video/output-%s.avi' % args.output_string, fourcc, 30.0, (width, height))

            txt_out = open('output/video/output-%s.txt' % "detect_face", 'w')

            frame_num = 1

            with open("output/video/video-det-fold-detect-face.txt", 'r') as f:
                bbox_line_list = f.read().splitlines()

            idx = 0
            while idx < len(bbox_line_list):
                line = bbox_line_list[idx]
                line = line.strip('\n')
                line = line.split(' ')
                det_frame_num = int(line[0])

                print(frame_num)

                # Stop at a certain frame number
                if frame_num > 1000:
                    break

                # Save all frames as they are if they don't have bbox annotation.
                while frame_num < det_frame_num:
                    ret, frame = video.read()
                    if ret == False:
                        out.release()
                        video.release()
                        txt_out.close()
                        sys.exit(0)
                    # out.write(frame)
                    frame_num += 1

                # Start processing frame with bounding box
                ret, frame = video.read()
                if ret == False:
                    break
                cv2_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                while True:
                    x_min, y_min, x_max, y_max = int(float(line[1])), int(float(line[2])), int(float(line[3])), int(
                        float(line[4]))

                    bbox_width = abs(x_max - x_min)
                    bbox_height = abs(y_max - y_min)
                    # x_min -= 3 * bbox_width / 4
                    # x_max += 3 * bbox_width / 4
                    # y_min -= 3 * bbox_height / 4
                    # y_max += bbox_height / 4
                    x_min -= 50
                    x_max += 50
                    y_min -= 50
                    y_max += 30
                    x_min = max(x_min, 0)
                    y_min = max(y_min, 0)
                    x_max = min(frame.shape[1], x_max)
                    y_max = min(frame.shape[0], y_max)
                    # Crop face loosely
                    img = cv2_frame[y_min:y_max, x_min:x_max] / 255
                    img = np.expand_dims(img, axis=0)

                    #print(img)
                    #print(img.shape)
                    yaw, pitch, roll = sess.run([yaw_predicted, pitch_predicted, roll_predicted], feed_dict={images_placeholder: img, istraining_placeholder:False})
                    #print(yaw, pitch, roll)

                    # print yaw_predicted, pitch_predicted,roll_predicted
                    # Print new frame with cube and axis
                    txt_out.write(str(frame_num) + ' %f %f %f\n' % (yaw, pitch, roll))
                    utils.plot_pose_cube(frame, yaw, pitch, roll, (x_min + x_max) / 2,
                                         (y_min + y_max) / 2, size=bbox_width)
                    # utils.draw_axis(frame, yaw_predicted, pitch_predicted, roll_predicted, tdx = (x_min + x_max) / 2, tdy= (y_min + y_max) / 2, size = bbox_height/2)
                    # Plot expanded bounding box
                    # cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0,255,0), 1)

                    # Peek next frame detection
                    next_frame_num = int(bbox_line_list[idx + 1].strip('\n').split(' ')[0])
                    # print 'next_frame_num ', next_frame_num
                    if next_frame_num == det_frame_num:
                        idx += 1
                        line = bbox_line_list[idx].strip('\n').split(' ')
                        det_frame_num = int(line[0])
                    else:
                        break

                idx += 1
                out.write(frame)
                frame_num += 1

            out.release()
            video.release()
            txt_out.close()

def run_gpu_degrad_utility_eval():

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True

    with tf.Graph().as_default():
        with tf.Session(config=config) as sess:
            istraining_placeholder = tf.placeholder(tf.bool)
            images_placeholder = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, None, None, 3))
            with tf.device('/gpu:%d' % 0):
                print('/gpu:%d' % 0)
                with tf.name_scope('%s_%d' % ('gpu', 0)) as scope:
                    X = tf.image.resize_bilinear(images_placeholder, [224, 224])
                    images_degraded = residualnet(X)
                    if FLAGS.use_avg_replicate:
                        images_degraded = _avg_replicate(images_degraded)

                    fc_yaw, fc_pitch, fc_roll = hopenet(images_degraded, is_training=istraining_placeholder)

                    idx_tensor = tf.expand_dims(tf.range(66, dtype=tf.float32), 0)
                    idx_tensor = tf.tile(idx_tensor, [FLAGS.batch_size, 1])

                    yaw_predicted = tf.reduce_sum(tf.multiply(tf.nn.softmax(fc_yaw, axis=1), idx_tensor),
                                                  axis=1) * 3 - 99
                    pitch_predicted = tf.reduce_sum(tf.multiply(tf.nn.softmax(fc_pitch, axis=1), idx_tensor),
                                                    axis=1) * 3 - 99
                    roll_predicted = tf.reduce_sum(tf.multiply(tf.nn.softmax(fc_roll, axis=1), idx_tensor),
                                                   axis=1) * 3 - 99

            init_op = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())
            sess.run(init_op)

            gvar_list = tf.global_variables()
            bn_moving_vars = [g for g in gvar_list if 'moving_mean' in g.name]
            bn_moving_vars += [g for g in gvar_list if 'moving_variance' in g.name]

            # for v in tf.trainable_variables()+bn_moving_vars:
            #     print("v.name: {}".format(v.name))
            #     print("v.shape: {}".format(sess.run(v).shape))
            #     print('=====================================')

            varlist = tf.trainable_variables() + bn_moving_vars
            saver = tf.train.Saver(varlist)
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir=FLAGS.checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Session restored from trained model at {}!'.format(ckpt.model_checkpoint_path))
            else:
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), FLAGS.checkpoint_dir)

            video = cv2.VideoCapture("output/video/videoplayback.mp4")

            # New cv2
            width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))  # float
            height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float
            #width, height = 224, 224

            # Define the codec and create VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            out = cv2.VideoWriter('output/video/output-%s.avi' % "detect_face", fourcc, 1, (width, height))

            # # Old cv2
            # width = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))   # float
            # height = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)) # float
            #
            # # Define the codec and create VideoWriter object
            # fourcc = cv2.cv.CV_FOURCC(*'MJPG')
            # out = cv2.VideoWriter('output/video/output-%s.avi' % args.output_string, fourcc, 30.0, (width, height))

            txt_out = open('output/video/output-%s.txt' % "detect_face", 'w')

            frame_num = 1

            with open("output/video/video-det-fold-detect-face.txt", 'r') as f:
                bbox_line_list = f.read().splitlines()

            idx = 0
            while idx < len(bbox_line_list):
                line = bbox_line_list[idx]
                line = line.strip('\n')
                line = line.split(' ')
                det_frame_num = int(line[0])

                print(frame_num)

                # Stop at a certain frame number
                if frame_num > 1000:
                    break

                # Save all frames as they are if they don't have bbox annotation.
                while frame_num < det_frame_num:
                    ret, frame = video.read()
                    if ret == False:
                        out.release()
                        video.release()
                        txt_out.close()
                        sys.exit(0)
                    # out.write(frame)
                    frame_num += 1

                # Start processing frame with bounding box
                ret, frame = video.read()
                if ret == False:
                    break
                cv2_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                while True:
                    x_min, y_min, x_max, y_max = int(float(line[1])), int(float(line[2])), int(float(line[3])), int(
                        float(line[4]))

                    bbox_width = abs(x_max - x_min)
                    bbox_height = abs(y_max - y_min)
                    # x_min -= 3 * bbox_width / 4
                    # x_max += 3 * bbox_width / 4
                    # y_min -= 3 * bbox_height / 4
                    # y_max += bbox_height / 4
                    x_min -= 50
                    x_max += 50
                    y_min -= 50
                    y_max += 30
                    x_min = max(x_min, 0)
                    y_min = max(y_min, 0)
                    x_max = min(frame.shape[1], x_max)
                    y_max = min(frame.shape[0], y_max)
                    # Crop face loosely
                    img = cv2_frame[y_min:y_max, x_min:x_max]
                    img = np.expand_dims(img, axis=0)

                    #print(img)
                    #print(img.shape)
                    img_degrad, yaw, pitch, roll = sess.run([images_degraded, yaw_predicted, pitch_predicted, roll_predicted], feed_dict={images_placeholder: img, istraining_placeholder:False})
                    #print(yaw, pitch, roll)

                    # print yaw_predicted, pitch_predicted,roll_predicted
                    # Print new frame with cube and axis
                    txt_out.write(str(frame_num) + ' %f %f %f\n' % (yaw, pitch, roll))
                    utils.plot_pose_cube(frame, yaw, pitch, roll, (x_min + x_max) / 2,
                                         (y_min + y_max) / 2, size=bbox_width)
                    # utils.draw_axis(frame, yaw_predicted, pitch_predicted, roll_predicted, tdx = (x_min + x_max) / 2, tdy= (y_min + y_max) / 2, size = bbox_height/2)
                    # Plot expanded bounding box
                    # cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0,255,0), 1)

                    # Peek next frame detection
                    next_frame_num = int(bbox_line_list[idx + 1].strip('\n').split(' ')[0])
                    # print 'next_frame_num ', next_frame_num
                    if next_frame_num == det_frame_num:
                        idx += 1
                        line = bbox_line_list[idx].strip('\n').split(' ')
                        det_frame_num = int(line[0])
                    else:
                        break

                idx += 1
                print(frame.shape)
                out.write(frame)
                frame_num += 1

            out.release()
            video.release()
            txt_out.close()

def main(_):
    #run_gpu_utility_eval()
    run_gpu_degrad_utility_eval()
    #run_validation_multi_model_mixed_restarting()

if __name__ == '__main__':
    tf.app.run()