#!/usr/bin/env python

'''
ECCV training method.
'''

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ["CUDA_VISIBLE_DEVICES"]="6"

import sys
sys.path.insert(0, '..')

import time
from six.moves import xrange
import input_data
import errno
import pprint
import itertools
from degradlNet import residualNet
from budgetNet import budgetNet
from utilityNet import utilityNet
from loss import *
from utils import *
from img_proc import _avg_replicate
import yaml
from tf_flags import FLAGS

from functions import placeholder_inputs, create_grad_accum_for_late_update, create_videos_reading_ops, create_summary_files
from bcolors import bcolors

try:
  xrange
except:
  xrange = range

def create_architecture_adversarial(cfg, multiplier_lst, logits_budget_lst_dct, loss_budget_lst_dct, scope, videos, utility_labels, budget_labels, dropout, is_training, lambda_):
    '''
    Create the architecture of the adversarial model in the graph
    '''
    # fd part:
    degrad_videos = residualNet(videos, is_video=True)
    fd_loss = tf.reduce_mean(tf.abs(degrad_videos - videos)) # L1_loss
    degrad_videos = _avg_replicate(degrad_videos) if FLAGS.use_avg_replicate else degrad_videos
    # fd part ends
    # fT part:
    logits_utility = utilityNet(degrad_videos, dropout, wd=0.001)
    loss_utility = tower_loss_xentropy_sparse(scope, logits_utility, utility_labels, use_weight_decay=True)
    # fT part ends
    # fb part:
    logits_budget = tf.zeros([cfg['TRAIN']['BATCH_SIZE'], cfg['DATA']['NUM_CLASSES_BUDGET']])
    loss_budget = 0.0
    budget_logits_lst = []
    for multiplier in multiplier_lst:
        print(multiplier)
        logits = budgetNet(degrad_videos, is_training, depth_multiplier=multiplier)
        budget_logits_lst.append(logits)
        loss = tower_loss_xentropy_sparse(scope, logits, budget_labels, use_weight_decay=False)
        logits_budget_lst_dct[str(multiplier)].append(logits)
        loss_budget_lst_dct[str(multiplier)].append(loss)
        logits_budget += logits / FLAGS.NBudget
        loss_budget += loss / FLAGS.NBudget
    # fd part ends.
    # Find the largest budget loss of the M ensembled budget models:
    if FLAGS.use_xentropy_uniform:
        max_adverse_budget_loss, argmax_adverse_budget_loss = tower_loss_max_xentropy_uniform(budget_logits_lst)
    else:
        max_adverse_budget_loss, argmax_adverse_budget_loss = tower_loss_max_neg_entropy(budget_logits_lst)
    # finish finding max_adverse_budget_loss and argmax_adverse_budget_loss.
    # calculate the total loss: L = LT+Lb
    if FLAGS.use_l1_loss:
        if lambda_ is None:
            loss_degrad = loss_utility + FLAGS._gamma * max_adverse_budget_loss
        else:
            loss_degrad = loss_utility + FLAGS._gamma * max_adverse_budget_loss + lambda_ * fd_loss
    else:
        loss_degrad = loss_utility + FLAGS._gamma * max_adverse_budget_loss
    # finish calculating the total loss.
    return loss_degrad, loss_budget, loss_utility, logits_budget, logits_utility, argmax_adverse_budget_loss


# Training set for traning, validation set for validation.
def run_adversarial_training(cfg, multiplier_lst, logits_budget_lst_dct, loss_budget_lst_dct):
    '''
    Algorithm 1 in the paper
    '''
    if not os.path.exists(FLAGS.ckpt_dir):
        os.makedirs(FLAGS.ckpt_dir)

    use_pretrained_model = True
    # define graph
    graph = tf.Graph()
    with graph.as_default():
        # global step:
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        # placeholder inputs:
        videos_placeholder, utility_labels_placeholder, budget_labels_placeholder, dropout_placeholder, _ = \
                                        placeholder_inputs(cfg['TRAIN']['BATCH_SIZE'] * cfg['TRAIN']['GPU_NUM'], cfg)
        is_training_placeholder = tf.placeholder(tf.bool)

        tower_grads_degrad, tower_grads_utility, tower_grads_budget = [], [], []

        # Compute Acc (fT, fb logits output)
        logits_utility_lst, logits_budget_lst = [], []

        # Compute Loss (LT, Lb_cross_entropy, Ld=LT+Lb_entropy?)
        loss_utility_lst, loss_budget_lst, loss_degrad_lst = [], [], []

        # Compute prediction with min entropy (most confident)
        # Use max uniform loss instead
        argmax_adverse_budget_loss_lst = []

        # Optimizer for the 3 components respectively
        opt_degrad = tf.train.AdamOptimizer(FLAGS.degradation_lr)
        opt_utility = tf.train.AdamOptimizer(FLAGS.utility_lr)
        opt_budget = tf.train.AdamOptimizer(FLAGS.budget_lr)

        if FLAGS.use_lambda_decay:
            _lambda_op = tf.train.exponential_decay(FLAGS._lambda, global_step=global_step, decay_steps=10, decay_rate=0.9)
        else:
            _lambda_op = tf.identity(FLAGS._lambda)

        with tf.variable_scope(tf.get_variable_scope()):
            for gpu_index in range(0, cfg['TRAIN']['GPU_NUM']):
                with tf.device('/gpu:%d' % gpu_index):
                    print('/gpu:%d' % gpu_index)
                    with tf.name_scope('%s_%d' % ('gpu', gpu_index)) as scope:
                        videos = videos_placeholder[gpu_index * cfg['TRAIN']['BATCH_SIZE']:(gpu_index + 1) * cfg['TRAIN']['BATCH_SIZE']]
                        utility_labels = utility_labels_placeholder[gpu_index * cfg['TRAIN']['BATCH_SIZE']:(gpu_index + 1) * cfg['TRAIN']['BATCH_SIZE']]
                        budget_labels = budget_labels_placeholder[gpu_index * cfg['TRAIN']['BATCH_SIZE']:(gpu_index + 1) * cfg['TRAIN']['BATCH_SIZE']]
                        loss_degrad, loss_budget, loss_utility, logits_budget, logits_utility, argmax_adverse_budget_loss = create_architecture_adversarial(cfg, multiplier_lst, logits_budget_lst_dct, loss_budget_lst_dct, scope, videos, utility_labels, budget_labels, dropout_placeholder, is_training_placeholder, _lambda_op)
                        loss_degrad_lst.append(loss_degrad)
                        loss_budget_lst.append(loss_budget)
                        loss_utility_lst.append(loss_utility)
                        logits_budget_lst.append(logits_budget)
                        logits_utility_lst.append(logits_utility)
                        argmax_adverse_budget_loss_lst.append(argmax_adverse_budget_loss)
                        varlist_degrad = [v for v in tf.trainable_variables() if any(x in v.name for x in ["DegradationModule"])]
                        varlist_utility = [v for v in tf.trainable_variables() if any(x in v.name for x in ["UtilityModule"])]
                        varlist_budget = [v for v in tf.trainable_variables() if any(x in v.name for x in ["BudgetModule"])]

                        grads_degrad = opt_degrad.compute_gradients(loss_degrad, varlist_degrad)
                        grads_budget = opt_budget.compute_gradients(loss_budget, varlist_budget)
                        grads_utility = opt_utility.compute_gradients(loss_utility, varlist_utility+varlist_degrad)

                        tower_grads_degrad.append(grads_degrad)
                        tower_grads_budget.append(grads_budget)
                        tower_grads_utility.append(grads_utility)

                        # Reuse variables for the next tower.
                        tf.get_variable_scope().reuse_variables()
        argmax_adverse_budget_loss_op = tf.concat(argmax_adverse_budget_loss_lst, 0)

        # Average losses over each GPU:
        loss_utility_op = tf.reduce_mean(loss_utility_lst, name='softmax')
        loss_budget_op = tf.reduce_mean(loss_budget_lst, name='softmax')
        loss_degrad_op = tf.reduce_mean(loss_degrad_lst, name='softmax')

        # Accuracy of utility:
        logits_utility = tf.concat(logits_utility_lst, 0)
        accuracy_util = accuracy(logits_utility, utility_labels_placeholder)

        # Accuracy of budget:
        logits_budget = tf.concat(logits_budget_lst, 0)
        accuracy_budget = accuracy(logits_budget, budget_labels_placeholder)

        acc_op_lst = []
        loss_op_lst = []
        for multiplier in multiplier_lst:
            acc_op = accuracy(tf.concat(logits_budget_lst_dct[str(multiplier)], 0), budget_labels_placeholder)
            acc_op_lst.append(acc_op)
            loss_op = tf.reduce_max(loss_budget_lst_dct[str(multiplier)])
            loss_op_lst.append(loss_op)

        zero_ops_degrad, accum_ops_degrad, apply_gradient_op_degrad = create_grad_accum_for_late_update(opt_degrad, tower_grads_degrad, varlist_degrad, global_step, decay_with_global_step=True)
        zero_ops_budget, accum_ops_budget, apply_gradient_op_budget = create_grad_accum_for_late_update(opt_budget, tower_grads_budget, varlist_budget, global_step, decay_with_global_step=False)
        zero_ops_utility, accum_ops_utility, apply_gradient_op_utility = create_grad_accum_for_late_update(opt_utility, tower_grads_utility, varlist_utility+varlist_degrad, global_step, decay_with_global_step=False)

        tr_videos_op, tr_action_labels_op, tr_actor_labels_op = create_videos_reading_ops(is_train=True, is_val=False, cfg=cfg)
        val_videos_op, val_action_labels_op, val_actor_labels_op = create_videos_reading_ops(is_train=False, is_val=True, cfg=cfg)

    # session config:
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True
    # run session:
    with tf.Session(graph=graph, config=config) as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        init_op = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())
        sess.run(init_op)

        # load ckpts: 
        if use_pretrained_model: # load ckpts from pretrained fd and fT.(By run_pretraining_degrad and run_pretraining_utility functions.)
            # fT and fd part:
            vardict_degradation = {v.name[:-2]: v for v in varlist_degrad}
            vardict_utility = {v.name[:-2]: v for v in varlist_utility}
            vardict = dict(vardict_degradation, **vardict_utility)
            restore_model_ckpt(sess, FLAGS.deg_target_models, vardict)
            # fb part:
            vardict_budget = {v.name[:-2]: v for v in varlist_budget}
            restore_model_ckpt(sess, FLAGS.budget_models, vardict_budget) # FLAGS.deg_target_models is the dir storing ckpt of theta_T and theta_d
        else: # load ckpts from previous training stage of this run_adversarial_training function.
            saver = tf.train.Saver(tf.trainable_variables())
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir=FLAGS.ckpt_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Session restored from trained model at {}!'.format(ckpt.model_checkpoint_path))
            else:
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), FLAGS.ckpt_dir)
        
        # saver and summary files:
        saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=10)
        loss_summary_file, validation_train_set_summary_file, validation_val_set_summary_file, model_restarting_summary_file = create_summary_files()

        # Adversarial training loop:
        for step in xrange(cfg['TRAIN']['TOP_MAXSTEP']):
            
            # Part 0:
            # Model initialization (only when step size=0) or model restarting
            if step == 0 or (FLAGS.use_restarting and step % FLAGS.restarting_step == 0):
                budget_varlist = [v for v in tf.trainable_variables() if any(x in v.name for x in ["BudgetModule"])]
                init_budget_op = tf.variables_initializer(budget_varlist)
                # reinitialize fb:
                sess.run(init_budget_op)
                # finish reinitializing fb
                # Train theta_B using Lb(X,Y_B) for FLAGS.retraining_step steps:
                for Restarting_step in range(0, cfg['TRAIN']['RETRAINING_STEP']):
                    start_time = time.time()
                    acc_util_lst, acc_budget_lst, loss_degrad_lst, loss_utility_lst, loss_budget_lst = [], [], [], [], []
                    sess.run(zero_ops_budget)
                    # accumulating gradient for late update:
                    for _ in itertools.repeat(None, 20):
                        tr_videos, tr_action_labels, tr_actor_labels = sess.run([tr_videos_op, tr_action_labels_op, tr_actor_labels_op])
                        _, acc_util, acc_budget, loss_degrad_value, loss_utility_value, loss_budget_value = sess.run([accum_ops_budget, accuracy_util, accuracy_budget, loss_degrad_op, loss_utility_op, loss_budget_op],
                            feed_dict={videos_placeholder: tr_videos,
                                        utility_labels_placeholder: tr_action_labels,
                                        budget_labels_placeholder: tr_actor_labels,
                                        dropout_placeholder: 1.0,
                                        is_training_placeholder: True,
                                        })
                        acc_util_lst.append(acc_util)
                        acc_budget_lst.append(acc_budget)
                        loss_degrad_lst.append(loss_degrad_value)
                        loss_utility_lst.append(loss_utility_value)
                        loss_budget_lst.append(loss_budget_value)
                    # finish accumulating gradient for late update
                    # after accumulating gradient, do the update on fb:
                    sess.run(apply_gradient_op_budget)
                    # finish update on fb
                    assert not np.isnan(np.mean(loss_degrad_lst)), 'Model diverged with loss = NaN'
                    loss_summary = 'Restarting (Budget), Step: {:4d}, Restarting_step: {:4d}, time: {:.4f}, budget loss: {:.8f}, ' \
                                    'training budget accuracy: {:.5f}, utility loss: {:.8f}, training utility accuracy: {:.5f},'.format(
                                    step, Restarting_step, time.time() - start_time, 
                                    np.mean(loss_budget_lst), np.mean(acc_budget_lst), np.mean(loss_utility_lst), np.mean(acc_util_lst))
                    model_restarting_summary_file.write(loss_summary + '\n')
                    print(loss_summary)
                # finish training theta_B using Lb(X,Y_B) for FLAGS.retraining_step steps.
            print('\n')
            # End part 0
            
            # Part 1:
            # Train f_d using L_T(X, Y_T)+L_b(X)' where L_b' is negative entropy or equivalently cross entropy with uniform labels.
            start_time = time.time()
            loss_degrad_lst, loss_utility_lst, loss_budget_lst = [], [], []
            sess.run(zero_ops_degrad)
            # accumulating gradient for late update:
            for _ in itertools.repeat(None, FLAGS.n_minibatches):
                tr_videos, tr_actor_labels, tr_action_labels = sess.run(
                                [tr_videos_op, tr_actor_labels_op, tr_action_labels_op])
                _, argmax_cent, loss_degrad_value, loss_utility_value, loss_budget_value = sess.run(
                                [accum_ops_degrad, argmax_adverse_budget_loss_op, loss_degrad_op, loss_utility_op, loss_budget_op],
                                feed_dict={
                                        videos_placeholder: tr_videos,
                                        utility_labels_placeholder: tr_action_labels,
                                        budget_labels_placeholder: tr_actor_labels,
                                        dropout_placeholder: 1.0,
                                        is_training_placeholder: True,})
                # print(argmax_cent)

                loss_degrad_lst.append(loss_degrad_value)
                loss_utility_lst.append(loss_utility_value)
                loss_budget_lst.append(loss_budget_value)
            # finish accumulating gradient for late update
            # after accumulating gradient, do the update on fd:
            _, _lambda = sess.run([apply_gradient_op_degrad, _lambda_op])
            # finish update on fd

            assert not np.isnan(np.mean(loss_degrad_lst)), 'Model diverged with loss = NaN'
            loss_summary = 'Alternating Training (Degradation), Lambda: {:.8f}, Step: {:4d}, time: {:.4f}, degrad loss: {:.8f}, utility loss: {:.8f}, budget loss: {:.8f}'.format(
                        _lambda, step, time.time() - start_time, np.mean(loss_degrad_lst), np.mean(loss_utility_lst), np.mean(loss_budget_lst))
            print(loss_summary)
            loss_summary_file.write(loss_summary + '\n')
            print('\n')
            # End part 1

            # Part 2: End-to-end train Ft and Fd using L_T
            if FLAGS.use_monitor_utility:
                for L_T_step in range(0, cfg['TRAIN']['L_T_MAXSTEP']):
                    start_time = time.time()
                    acc_util_lst, acc_budget_lst, loss_value_lst, loss_utility_lst, loss_budget_lst = [], [], [], [], []
                    # validate the performance on target task (acc_util_lst and loss_utility_lst)
                    for _ in itertools.repeat(None, FLAGS.n_minibatches):
                        val_videos, val_action_labels, val_actor_labels = sess.run([val_videos_op, val_action_labels_op, val_actor_labels_op])
                        acc_util, acc_budget, loss_value, loss_utility, loss_budget = sess.run(
                                        [accuracy_util, accuracy_budget, loss_op, loss_utility_op, loss_budget_op],
                                        feed_dict={videos_placeholder: val_videos,
                                            utility_labels_placeholder: val_action_labels,
                                            budget_labels_placeholder: val_actor_labels,
                                            dropout_placeholder: 1.0,
                                            is_training_placeholder: True,
                                        })
                        acc_util_lst.append(acc_util)
                        acc_budget_lst.append(acc_budget)
                        loss_value_lst.append(loss_value)
                        loss_utility_lst.append(loss_utility)
                        loss_budget_lst.append(loss_budget)
                    # finish calculating acc_util_lst and loss_utility_lst
                    # print the validation result:
                    val_summary = "Monitoring L_T:\n" \
                            "Step: {:4d}, L_T_step: {:4d}, time: {:.4f}, total loss: {:.8f}, utility loss: {:.8f}, budget loss: {:.8f}, " \
                            "validation utility accuracy: {:.5f}, validation budget accuracy: {:.5f}" .format(
                            step, L_T_step, 
                            time.time() - start_time, np.mean(loss_value_lst),
                            np.mean(loss_utility_lst), np.mean(loss_budget_lst),
                            np.mean(acc_util_lst), np.mean(acc_budget_lst))
                    print(val_summary)

                    # breaking condition: (if performance on L_T is still good)
                    if np.mean(acc_util_lst) >= FLAGS.highest_util_acc_val:
                        print(bcolors.OKGREEN  + 'pass utility acc bar!\n' + bcolors.ENDC)
                        break
                    
                    start_time = time.time()
                    sess.run(zero_ops_utility)
                    acc_util_lst, acc_budget_lst, loss_degrad_lst, loss_utility_lst, loss_budget_lst = [], [], [], [], []
                    # accumulating gradient for late update:
                    for _ in itertools.repeat(None, FLAGS.n_minibatches):
                        tr_videos, tr_action_labels, tr_actor_labels = sess.run([tr_videos_op, tr_action_labels_op, tr_actor_labels_op])
                        _, acc_util, acc_budget, loss_degrad_value, loss_utility_value, loss_budget_value = sess.run([accum_ops_utility, accuracy_util, accuracy_budget, loss_degrad_op, loss_utility_op, loss_budget_op],
                                                feed_dict={videos_placeholder: tr_videos,
                                                            utility_labels_placeholder: tr_action_labels,
                                                            budget_labels_placeholder: tr_actor_labels,
                                                            dropout_placeholder: 0.5,
                                                            is_training_placeholder: True,
                                                            })
                        acc_util_lst.append(acc_util)
                        acc_budget_lst.append(acc_budget)
                        loss_degrad_lst.append(loss_degrad_value)
                        loss_utility_lst.append(loss_utility_value)
                        loss_budget_lst.append(loss_budget_value)
                    # finish accumulating gradient for late update
                    # after accumulating gradient, do the update on fT and fd:
                    sess.run([apply_gradient_op_utility])
                    # finish update on fT and fd
                    assert not np.isnan(np.mean(loss_degrad_lst)), 'Model diverged with loss = NaN'
                    loss_summary = 'Alternating Training (Utility), Step: {:4d}, L_T_step: {:4d}, time: {:.4f}, ' \
                                'degrad loss: {:.8f}, utility loss: {:.8f}, budget loss: {:.8f}, '.format(
                                step, L_T_step, time.time() - start_time, np.mean(loss_degrad_lst),
                                np.mean(loss_utility_lst), np.mean(loss_budget_lst))

                    print(loss_summary)
                print('\n')
            # End part 2

            # Part 3: train Fb using L_b (cross entropy)
            if FLAGS.use_monitor_budget:
                for L_b_step in range(0, cfg['TRAIN']['L_B_MAXSTEP']):
                    start_time = time.time()
                    sess.run(zero_ops_budget)
                    acc_util_lst, acc_budget_lst, loss_degrad_lst, loss_utility_lst, loss_budget_lst = [], [], [], [], []
                    # accumulating gradient for late update:
                    for _ in itertools.repeat(None, FLAGS.n_minibatches):
                        tr_videos, tr_action_labels, tr_actor_labels = sess.run([tr_videos_op, tr_action_labels_op, tr_actor_labels_op])

                        _, acc_util, acc_budget, loss_degrad_value, loss_utility_value, loss_budget_value = sess.run(
                                    [accum_ops_budget, accuracy_util, accuracy_budget, loss_degrad_op, loss_utility_op, loss_budget_op],
                                    feed_dict={
                                        videos_placeholder: tr_videos,
                                        utility_labels_placeholder: tr_action_labels,
                                        budget_labels_placeholder: tr_actor_labels,
                                        dropout_placeholder: 1.0,
                                        is_training_placeholder: True,})

                        acc_util_lst.append(acc_util)
                        acc_budget_lst.append(acc_budget)
                        loss_degrad_lst.append(loss_degrad_value)
                        loss_utility_lst.append(loss_utility_value)
                        loss_budget_lst.append(loss_budget_value)
                    # finish accumulating gradient for late update
                    # after accumulating gradient, do the update on fb:
                    sess.run([apply_gradient_op_budget])
                    # finish update on fb
                    
                    assert not np.isnan(np.mean(loss_degrad_lst)), 'Model diverged with loss = NaN'
                    loss_summary = 'Alternating Training (Budget), Step: {:4d}, L_b_step: {:4d}, time: {:.4f}, degrad loss: {:.8f}, ' \
                                'utility loss: {:.8f}, budget loss: {:.8f}, training utility accuracy: {:.5f}, training budget accuracy: {:.5f}'.format(
                                step, L_b_step, time.time() - start_time, np.mean(loss_degrad_lst), 
                                np.mean(loss_utility_lst), np.mean(loss_budget_lst), np.mean(acc_util_lst), np.mean(acc_budget_lst))

                    print(loss_summary)

                    # # breaking condition:
                    # if np.mean(acc_budget_lst) >= FLAGS.highest_budget_acc_train:
                    #     break
                print('\n')
            # End part 3

            # Do evaluation:
            if step % cfg['TRAIN']['VAL_STEP'] == 0:
                start_time = time.time()
                acc_util_train_lst, acc_budget_train_lst, loss_degrad_train_lst, loss_utility_train_lst, loss_budget_train_lst = [], [], [], [], []
                acc_util_val_lst, acc_budget_val_lst, loss_degrad_val_lst, loss_utility_val_lst, loss_budget_val_lst = [], [], [], [], []
                for _ in itertools.repeat(None, FLAGS.n_minibatches):
                    tr_videos, tr_action_labels, tr_actor_labels = sess.run(
                                [tr_videos_op, tr_action_labels_op, tr_actor_labels_op])
                    acc_util, acc_budget, loss_degrad_value, loss_utility_value, loss_budget_value = sess.run([accuracy_util, accuracy_budget,
                                                                                            loss_degrad_op, loss_utility_op, loss_budget_op],
                                                        feed_dict={videos_placeholder: tr_videos,
                                                                    utility_labels_placeholder: tr_action_labels,
                                                                    budget_labels_placeholder: tr_actor_labels,
                                                                    dropout_placeholder: 1.0,
                                                                    is_training_placeholder: True,
                                                                    })
                    acc_util_train_lst.append(acc_util)
                    acc_budget_train_lst.append(acc_budget)
                    loss_degrad_train_lst.append(loss_degrad_value)
                    loss_utility_train_lst.append(loss_utility_value)
                    loss_budget_train_lst.append(loss_budget_value)

                validation_train_set_summary = "Evaluation validation_train_set_summary\n" \
                        "Step: {:4d}, time: {:.4f}, degrad loss: {:.8f}, utility loss: {:.8f}, budget loss: {:.8f}, training utility accuracy: {:.5f}, training budget accuracy: {:.5f}".format(
                        step, time.time() - start_time, np.mean(loss_degrad_train_lst),
                        np.mean(loss_utility_train_lst), np.mean(loss_budget_train_lst),
                        np.mean(acc_util_train_lst), np.mean(acc_budget_train_lst))
                print(validation_train_set_summary)
                print('\n')
                validation_train_set_summary_file.write(validation_train_set_summary + '\n')

                for _ in itertools.repeat(None, FLAGS.n_minibatches_eval):
                    val_videos, val_action_labels, val_actor_labels = sess.run(
                                    [val_videos_op, val_action_labels_op, val_actor_labels_op])
                    acc_util, acc_budget, loss_degrad_value, loss_utility_value, loss_budget_value = sess.run([accuracy_util, accuracy_budget,
                                                                                            loss_degrad_op, loss_utility_op, loss_budget_op],
                                                        feed_dict={videos_placeholder: val_videos,
                                                                    utility_labels_placeholder: val_action_labels,
                                                                    budget_labels_placeholder: val_actor_labels,
                                                                    dropout_placeholder: 1.0,
                                                                    is_training_placeholder: True,
                                                                    })
                    acc_util_val_lst.append(acc_util)
                    acc_budget_val_lst.append(acc_budget)
                    loss_degrad_val_lst.append(loss_degrad_value)
                    loss_utility_val_lst.append(loss_utility_value)
                    loss_budget_val_lst.append(loss_budget_value)

                validation_val_set_summary = "Evaluation validation_val_set_summary\n" \
                        "Step: {:4d}, time: {:.4f}, degrad loss: {:.8f}, utility loss: {:.8f}, budget loss: {:.8f}, validation utility accuracy: {:.5f}, validation budget accuracy: {:.5f}".format(
                        step, time.time() - start_time, np.mean(loss_degrad_val_lst),
                        np.mean(loss_utility_val_lst), np.mean(loss_budget_val_lst),
                        np.mean(acc_util_val_lst), np.mean(acc_budget_val_lst))
                print(validation_val_set_summary)
                print('\n')
                validation_val_set_summary_file.write(validation_val_set_summary)
            # End evaluation
            # Save ckpt for kb_adversarial learning:
            if step % cfg['TRAIN']['SAVE_STEP'] == 0 or (step + 1) == cfg['TRAIN']['TOP_MAXSTEP']:
                checkpoint_path = os.path.join(FLAGS.ckpt_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
            # End evaluation

        loss_summary_file.close()
        validation_train_set_summary_file.close()
        validation_val_set_summary_file.close()
        coord.request_stop()
        coord.join(threads)
    print("done")


def main(_):
    # config:
    cfg = yaml.load(open('params.yml'))
    pp = pprint.PrettyPrinter()
    pp.pprint(FLAGS.__flags)
    pp.pprint(cfg)

    # The depth multiplier list for creating different budget models ensemble
    multiplier_lst = [0.60 - i * 0.02 for i in range(FLAGS.NBudget)]

    # The dict of logits for each different budget model to get accuracy
    logits_budget_lst_dct = {str(multiplier): [] for multiplier in multiplier_lst}
    loss_budget_lst_dct = {str(multiplier): [] for multiplier in multiplier_lst}

    run_adversarial_training(cfg, multiplier_lst, logits_budget_lst_dct, loss_budget_lst_dct)

if __name__ == '__main__':
    tf.app.run()
