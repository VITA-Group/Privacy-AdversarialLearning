#!/usr/bin/env python

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ["CUDA_VISIBLE_DEVICES"]=""

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

try:
  xrange
except:
  xrange = range

class AdversarialTraining(object):

    def __init__(self, cfg):
        self.sess = None
        self.cfg = cfg
        # The depth multiplier list for creating different budget models ensemble
        self.multiplier_lst = [0.60 - i * 0.02 for i in range(FLAGS.NBudget)]

        # The dict of logits for each different budget model to get accuracy
        self.logits_budget_lst_dct = {str(multiplier): [] for multiplier in self.multiplier_lst}
        self.loss_budget_lst_dct = {str(multiplier): [] for multiplier in self.multiplier_lst}

        # Global step for the exponential decay on lambda (controlling the relative weight of L1 loss)
        self.global_step = None

    '''
    Create the placeholder ops in the graph
    '''
    def placeholder_inputs(self, batch_size):
        videos_placeholder = tf.placeholder(tf.float32,
                                            shape=(batch_size, self.cfg['DATA']['DEPTH'], 112, 112, self.cfg['DATA']['NCHANNEL']))
        action_labels_placeholder = tf.placeholder(tf.int64, shape=(batch_size))
        actor_labels_placeholder = tf.placeholder(tf.int64, shape=(batch_size))
        dropout_placeholder = tf.placeholder(tf.float32)
        isTraining_placeholder = tf.placeholder(tf.bool)
        return videos_placeholder, action_labels_placeholder, actor_labels_placeholder, dropout_placeholder, isTraining_placeholder

    def create_architecture_degrad(self, scope, images, labels):
        degrad_images = residualNet(images, is_video=False)
        loss = tower_loss_mse(scope, degrad_images, labels)
        return loss
    
    def create_architecture_pretraining(self, scope, videos, utility_labels, dropout):
        degrad_videos = residualNet(videos, is_video=True)
        degrad_videos = _avg_replicate(degrad_videos) if FLAGS.use_avg_replicate else degrad_videos

        logits_utility = utilityNet(degrad_videos, dropout, wd=0.001)
        loss_utility = tower_loss_xentropy_sparse(scope, logits_utility, utility_labels, use_weight_decay=True)

        return loss_utility, logits_utility

    '''
    Create the architecture of the adversarial model in the graph
    '''
    def create_architecture_adversarial(self, scope, videos, utility_labels, budget_labels, dropout, is_training, batch_size, lambda_):
        degrad_videos = residualNet(videos, is_video=True)
        L1_loss = tf.reduce_mean(tf.abs(degrad_videos - videos))
        degrad_videos = _avg_replicate(degrad_videos) if FLAGS.use_avg_replicate else degrad_videos

        logits_utility = utilityNet(degrad_videos, dropout, wd=0.001)
        loss_utility = tower_loss_xentropy_sparse(scope, logits_utility, utility_labels, use_weight_decay=True)
        logits_budget = tf.zeros([batch_size, self.cfg['DATA']['NUM_CLASSES_BUDGET']])
        loss_budget = 0.0
        logits_lst = []
        for multiplier in self.multiplier_lst:
            print(multiplier)
            logits = budgetNet(degrad_videos, is_training, depth_multiplier=multiplier)
            logits_lst.append(logits)
            loss = tower_loss_xentropy_sparse(scope, logits, budget_labels, use_weight_decay=False)
            self.logits_budget_lst_dct[str(multiplier)].append(logits)
            self.loss_budget_lst_dct[str(multiplier)].append(loss)
            logits_budget += logits / FLAGS.NBudget
            loss_budget += loss / FLAGS.NBudget

        if FLAGS.use_xentropy_uniform:
            max_adverse_budget_loss, argmax_adverse_budget_loss = tower_loss_max_xentropy_uniform(logits_lst)
        else:
            max_adverse_budget_loss, argmax_adverse_budget_loss = tower_loss_max_neg_entropy(logits_lst)

        if FLAGS.use_l1_loss:
            if lambda_ is None:
                loss_degrad = loss_utility + FLAGS._gamma * max_adverse_budget_loss
            else:
                loss_degrad = loss_utility + FLAGS._gamma * max_adverse_budget_loss + lambda_ * L1_loss
        else:
            loss_degrad = loss_utility + FLAGS._gamma_ * max_adverse_budget_loss

        return loss_degrad, loss_budget, loss_utility, logits_budget, logits_utility, argmax_adverse_budget_loss

    '''
    Late update: (accumulate gradients and late update the parameters)
    '''
    def create_grad_accum_for_late_update(self, opt, tower_grads, tvarlist, decay_with_global_step=False):
        # Average grads over GPUs (towers):
        grads = average_gradients(tower_grads)

        # accum_vars are variable storing the accumulated gradient
        # zero_ops are used to zero the accumulated gradient (comparable to optimizer.zero_grad() in PyTorch)
        with tf.device('/cpu:%d' % 0):
            accum_vars = [tf.Variable(tf.zeros_like(tvar.initialized_value()), trainable=False) for tvar in
                                     tvarlist]
            zero_ops = [tvar.assign(tf.zeros_like(tvar)) for tvar in accum_vars]

        accum_ops = [accum_vars[i].assign_add(gv[0] / FLAGS.n_minibatches) for i, gv in enumerate(grads)]

        if decay_with_global_step:
            global_increment = self.global_step.assign_add(1)
            with tf.control_dependencies([global_increment]):
                apply_gradient_op = opt.apply_gradients(
                    [(accum_vars[i].value(), gv[1]) for i, gv in enumerate(grads)], global_step=None)
        else:
            apply_gradient_op = opt.apply_gradients(
                [(accum_vars[i].value(), gv[1]) for i, gv in enumerate(grads)], global_step=None)


        return zero_ops, accum_ops, apply_gradient_op

    '''
    Multi-thread data fetching from queue
    '''
    def create_videos_reading_ops(self, is_train, is_val):
        train_files = [os.path.join(self.cfg['DATA']['TRAIN_FILES_DIR'], f) for f in
                       os.listdir(self.cfg['DATA']['TRAIN_FILES_DIR']) if f.endswith('.tfrecords')]
        val_files = [os.path.join(self.cfg['DATA']['VAL_FILES_DIR'], f) for f in
                     os.listdir(self.cfg['DATA']['VAL_FILES_DIR']) if f.endswith('.tfrecords')]
        test_files = [os.path.join(self.cfg['DATA']['TEST_FILES_DIR'], f) for f in
                      os.listdir(self.cfg['DATA']['TEST_FILES_DIR']) if f.endswith('.tfrecords')]

        num_threads = self.cfg['DATA']['NUM_THREADS']
        num_examples_per_epoch = self.cfg['TRAIN']['NUM_EXAMPLES_PER_EPOCH']
        if is_train:
            batch_size = self.cfg['TRAIN']['BATCH_SIZE'] * self.cfg['TRAIN']['GPU_NUM']
            videos_op, action_labels_op, actor_labels_op = input_data.inputs_videos(filenames=train_files,
                                                                                    batch_size=batch_size,
                                                                                    num_epochs=None,
                                                                                    num_threads=num_threads,
                                                                                    num_examples_per_epoch=num_examples_per_epoch,
                                                                                    shuffle=True)
        elif is_val:
            batch_size = self.cfg['TRAIN']['BATCH_SIZE'] * self.cfg['TRAIN']['GPU_NUM']
            videos_op, action_labels_op, actor_labels_op = input_data.inputs_videos(filenames=val_files,
                                                                                    batch_size=batch_size,
                                                                                    num_epochs=None,
                                                                                    num_threads=num_threads,
                                                                                    num_examples_per_epoch=num_examples_per_epoch,
                                                                                    shuffle=True)

        else:
            batch_size = self.cfg['TEST']['BATCH_SIZE'] * self.cfg['TEST']['GPU_NUM']
            videos_op, action_labels_op, actor_labels_op = input_data.inputs_videos(filenames=test_files,
                                                                                    batch_size=batch_size,
                                                                                    num_epochs=1,
                                                                                    num_threads=num_threads,
                                                                                    num_examples_per_epoch=num_examples_per_epoch,
                                                                                    shuffle=False)

        return videos_op, action_labels_op, actor_labels_op

    def create_images_reading_ops(self, is_train, is_val):
        train_files = [os.path.join(self.cfg['DATA']['TRAIN_FILES_DEG_DIR'], f) for f in
                       os.listdir(self.cfg['DATA']['TRAIN_FILES_DEG_DIR']) if f.endswith('.tfrecords')]
        val_files = [os.path.join(self.cfg['DATA']['VAL_FILES_DEG_DIR'], f) for f in
                     os.listdir(self.cfg['DATA']['VAL_FILES_DEG_DIR']) if f.endswith('.tfrecords')]

        num_threads = self.cfg['DATA']['NUM_THREADS']
        num_examples_per_epoch = self.cfg['TRAIN']['NUM_EXAMPLES_PER_EPOCH']
        batch_size = self.cfg['TRAIN']['BATCH_SIZE'] * self.cfg['TRAIN']['GPU_NUM']

        if is_train:
            images_op, labels_op = input_data.inputs_images(filenames=train_files,
                                                            batch_size=batch_size,
                                                            num_epochs=None,
                                                            num_threads=num_threads,
                                                            num_examples_per_epoch=num_examples_per_epoch)
        if is_val:
            images_op, labels_op = input_data.inputs_images(filenames=val_files,
                                                            batch_size=batch_size,
                                                            num_epochs=None,
                                                            num_threads=num_threads,
                                                            num_examples_per_epoch=num_examples_per_epoch)
        return images_op, labels_op

    '''
    Creating summary files
    '''
    def create_summary_files(self):
        if not os.path.exists(FLAGS.summary_dir):
            os.makedirs(FLAGS.summary_dir)
        loss_summary_file = open(FLAGS.summary_dir + 'loss_summary.txt', 'w')
        train_summary_file = open(FLAGS.summary_dir + 'train_summary.txt', 'w')
        test_summary_file = open(FLAGS.summary_dir + 'test_summary.txt', 'w')
        model_restarting_summary_file = open(FLAGS.summary_dir + 'model_summary.txt', 'w')
        return loss_summary_file, train_summary_file, test_summary_file, model_restarting_summary_file

    '''
    Initialize f_d as identity mapping of the input
    '''
    def run_pretraining_degrad(self):
        if not os.path.exists(FLAGS.degradation_models):
            os.makedirs(FLAGS.degradation_models)
        start_from_trained_model = True

        graph = tf.Graph()
        with graph.as_default():
            batch_size = self.cfg['TRAIN']['BATCH_SIZE'] * self.cfg['TRAIN']['GPU_NUM']
            images_placeholder = tf.placeholder(tf.float32, [batch_size, 120, 160, 3], name='images')
            labels_placeholder = tf.placeholder(tf.float32, [batch_size, 120, 160, 3], name='labels')
            tower_grads = []
            losses = []
            opt = tf.train.AdamOptimizer(1e-3)

            with tf.variable_scope(tf.get_variable_scope()) as scope:
                for gpu_index in range(0, self.cfg['TRAIN']['GPU_NUM']):
                    with tf.device('/gpu:%d' % gpu_index):
                        print('/gpu:%d' % gpu_index)
                        with tf.name_scope('%s_%d' % ('gpu', gpu_index)) as scope:
                            images = images_placeholder[gpu_index * self.cfg['TRAIN']['BATCH_SIZE']:(gpu_index + 1) * self.cfg['TRAIN']['BATCH_SIZE']]
                            labels = labels_placeholder[gpu_index * self.cfg['TRAIN']['BATCH_SIZE']:(gpu_index + 1) * self.cfg['TRAIN']['BATCH_SIZE']]
                            loss = self.create_architecture_degrad(scope, images, labels)
                            losses.append(loss)
                            grads = opt.compute_gradients(loss)
                            tower_grads.append(grads)
                            tf.get_variable_scope().reuse_variables()
            
            loss_op = tf.reduce_mean(losses, name='mse')
            psnr_op = tf.multiply(tf.constant(20, dtype=tf.float32),
                                  tf.log(1 / tf.sqrt(loss_op)) / tf.log(tf.constant(10, dtype=tf.float32)), name='psnr')

            tf.summary.scalar('loss', loss_op)

            grads_avg = average_gradients(tower_grads)
            train_op = opt.apply_gradients(grads_avg)


            tr_images_op, tr_labels_op = self.create_images_reading_ops(is_train=True, is_val=False)
            val_images_op, val_labels_op = self.create_images_reading_ops(is_train=False, is_val=True)

            conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
            conf.gpu_options.allow_growth = True

            self.sess = tf.Session(graph=graph, config=conf)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)
            init_op = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())
            self.sess.run(init_op)

            if start_from_trained_model:
                vardict = {v.name[:-2]: v for v in tf.trainable_variables()}
                saver = tf.train.Saver(vardict)
                print(vardict)
                ckpt = tf.train.get_checkpoint_state(checkpoint_dir=FLAGS.degradation_models)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(self.sess, ckpt.model_checkpoint_path)
                    print('Session Restored!')
                else:
                    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), FLAGS.degradation_models)

            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(FLAGS.log_dir + 'train', self.sess.graph)
            val_writer = tf.summary.FileWriter(FLAGS.log_dir + 'val', self.sess.graph)

            print("Training...")
            saver = tf.train.Saver(tf.trainable_variables())
            for step in range(self.cfg['TRAIN']['MAX_STEPS']):
                # Run by batch images
                start_time = time.time()
                tr_images, tr_labels = self.sess.run([tr_images_op, tr_labels_op])
                print(tr_images.shape)
                print(tr_labels.shape)
                tr_feed = {images_placeholder: tr_images, labels_placeholder: tr_labels}
                _, loss_value = self.sess.run([train_op, loss_op], feed_dict=tr_feed)
                assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
                print("Step: [%2d], time: [%4.4f], training_loss = [%.8f]" % (step, time.time() - start_time, loss_value))
                if step % self.cfg['TRAIN']['VAL_STEP'] == 0:
                    val_images, val_labels = self.sess.run([val_images_op, val_labels_op])
                    val_feed = {images_placeholder: val_images, labels_placeholder: val_labels}
                    summary, loss_value, psnr = self.sess.run([merged, loss_op, psnr_op], feed_dict=val_feed)
                    print("Step: [%2d], time: [%4.4f], validation_loss = [%.8f], validation_psnr = [%.8f]" %
                          (step, time.time() - start_time, loss_value, psnr))
                    val_writer.add_summary(summary, step)
                    tr_images, tr_labels = self.sess.run([tr_images_op, tr_labels_op])
                    tr_feed = {images_placeholder: tr_images, labels_placeholder: tr_labels}
                    summary, loss_value, psnr = self.sess.run([merged, loss_op, psnr_op], feed_dict=tr_feed)
                    print("Step: [%2d], time: [%4.4f], training_loss = [%.8f], training_psnr = [%.8f]" %
                          (step, time.time() - start_time, loss_value, psnr))
                    train_writer.add_summary(summary, step)
                if step % self.cfg['TRAIN']['SAVE_STEP'] == 0 or (step + 1) == self.cfg['TRAIN']['MAX_STEPS']:
                    checkpoint_path = os.path.join(FLAGS.degradation_models, 'model.ckpt')
                    saver.save(self.sess, checkpoint_path, global_step=step)

            coord.request_stop()
            coord.join(threads)
            self.sess.close()

    '''
    Initialize f_T on pretrained f_d 
    '''
    def run_pretraining_utility(self):
        if not os.path.exists(FLAGS.whole_pretraining):
            os.makedirs(FLAGS.whole_pretraining)

        use_pretrained_model = True
        graph = tf.Graph()
        with graph.as_default():
            self.global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
            videos_placeholder, utility_labels_placeholder, budget_labels_placeholder, dropout_placeholder, _ = \
                                            self.placeholder_inputs(self.cfg['TRAIN']['BATCH_SIZE'] * self.cfg['TRAIN']['GPU_NUM'])
            tower_grads_degrad, tower_grads_utility_main, tower_grads_utility_finetune = [], [], []

            # Compute Acc
            logits_utility_lst = []

            # Compute Loss
            loss_utility_lst = []

            opt_degrad = tf.train.AdamOptimizer(1e-3)
            opt_utility_finetune = tf.train.AdamOptimizer(1e-4)
            opt_utility_main = tf.train.AdamOptimizer(1e-5)

            with tf.variable_scope(tf.get_variable_scope()) as scope:
                for gpu_index in range(0, self.cfg['TRAIN']['GPU_NUM']):
                    with tf.device('/gpu:%d' % gpu_index):
                        print('/gpu:%d' % gpu_index)
                        with tf.name_scope('%s_%d' % ('gpu', gpu_index)) as scope:
                            videos = videos_placeholder[gpu_index * self.cfg['TRAIN']['BATCH_SIZE']:(gpu_index + 1) * self.cfg['TRAIN']['BATCH_SIZE']]
                            utility_labels = utility_labels_placeholder[gpu_index * self.cfg['TRAIN']['BATCH_SIZE']:(gpu_index + 1) * self.cfg['TRAIN']['BATCH_SIZE']]

                            loss_utility, logits_utility = self.create_architecture_pretraining(scope, videos, utility_labels, dropout_placeholder)
                            logits_utility_lst.append(logits_utility)
                            loss_utility_lst.append(loss_utility)


                            varlist_degrad = [v for v in tf.trainable_variables() if
                                                            any(x in v.name for x in ["DegradationModule"])]
                            varlist_utility = [v for v in tf.trainable_variables() if
                                                            any(x in v.name for x in ["UtilityModule"])]
                            varlist_utility_finetune = [v for v in varlist_utility if
                                                            any(x in v.name.split('/')[1] for x in ["out", "d2"])]
                            varlist_utility_main = list(set(varlist_utility) - set(varlist_utility_finetune))

                            grads_degrad = opt_degrad.compute_gradients(loss_utility, varlist_degrad)
                            grads_utility_main = opt_utility_main.compute_gradients(loss_utility, varlist_utility_main)
                            grads_utility_finetune = opt_utility_finetune.compute_gradients(loss_utility, varlist_utility_finetune)

                            tower_grads_degrad.append(grads_degrad)
                            tower_grads_utility_main.append(grads_utility_main)
                            tower_grads_utility_finetune.append(grads_utility_finetune)

                            # Reuse variables for the next tower.
                            tf.get_variable_scope().reuse_variables()

            loss_utility_op = tf.reduce_mean(loss_utility_lst, name='softmax')

            logits_utility_op = tf.concat(logits_utility_lst, 0)
            accuracy_util = accuracy(logits_utility_op, utility_labels_placeholder)

            zero_ops_degrad, accum_ops_degrad, apply_gradient_op_degrad = \
                    self.create_grad_accum_for_late_update(opt_degrad, tower_grads_degrad, varlist_degrad, decay_with_global_step=False)
            zero_ops_utility_finetune, accum_ops_utility_finetune, apply_gradient_op_utility_finetune = \
                    self.create_grad_accum_for_late_update(opt_utility_finetune, tower_grads_utility_finetune, varlist_utility_finetune, decay_with_global_step=False)
            zero_ops_utility_main, accum_ops_utility_main, apply_gradient_op_utility_main = \
                    self.create_grad_accum_for_late_update(opt_utility_main, tower_grads_utility_main, varlist_utility_main, decay_with_global_step=False)

            tr_videos_op, tr_action_labels_op, tr_actor_labels_op = self.create_videos_reading_ops(is_train=True, is_val=False)
            val_videos_op, val_action_labels_op, val_actor_labels_op = self.create_videos_reading_ops(is_train=False, is_val=True)

            config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(graph=graph, config=config)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)

            init_op = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())
            self.sess.run(init_op)

            # Create a saver for writing training checkpoints.
            if use_pretrained_model:
                restore_model_ckpt(self.sess, FLAGS.degradation_models, varlist_degrad, "DegradationModule")
                restore_model_pretrained_C3D(self.sess, self.cfg)
            else:
                saver = tf.train.Saver(tf.trainable_variables())
                ckpt = tf.train.get_checkpoint_state(checkpoint_dir=FLAGS.whole_pretraining)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(self.sess, ckpt.model_checkpoint_path)
                    print('Session restored from trained model at {}!'.format(ckpt.model_checkpoint_path))
                else:
                    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), FLAGS.whole_pretraining)

            saver = tf.train.Saver(tf.trainable_variables())
            for step in xrange(500):
                start_time = time.time()
                self.sess.run([zero_ops_utility_finetune, zero_ops_utility_main, zero_ops_degrad])
                loss_utility_lst = []
                for _ in itertools.repeat(None, FLAGS.n_minibatches):
                    tr_videos, tr_videos_labels = self.sess.run([tr_videos_op, tr_action_labels_op])
                    _, _, loss_utility = self.sess.run(
                            [accum_ops_utility_finetune, accum_ops_utility_main, loss_utility_op],
                            feed_dict={videos_placeholder: tr_videos,
                                       utility_labels_placeholder: tr_videos_labels,
                                       dropout_placeholder: 1.0,
                                       })
                    loss_utility_lst.append(loss_utility)
                self.sess.run([apply_gradient_op_utility_finetune, apply_gradient_op_utility_main,
                              apply_gradient_op_degrad])
                loss_summary = 'Utility Module + Degradation Module, Step: {:4d}, time: {:.4f}, utility loss: {:.8f}'.format(
                        step,
                        time.time() - start_time,
                        np.mean(loss_utility_lst))
                print(loss_summary)

                if step % self.cfg['TRAIN']['VAL_STEP'] == 0:
                    start_time = time.time()
                    acc_util_train_lst, loss_utility_train_lst = [], []
                    for _ in itertools.repeat(None, 30):
                        tr_videos, tr_videos_labels = self.sess.run([tr_videos_op, tr_action_labels_op])
                        acc_util, loss_utility = self.sess.run([accuracy_util, loss_utility_op],
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
                    acc_util_val_lst, loss_utility_val_lst = [], []
                    for _ in itertools.repeat(None, 30):
                        val_videos, val_videos_labels = self.sess.run([val_videos_op, val_action_labels_op])
                        acc_util, loss_utility = self.sess.run([accuracy_util, loss_utility_op],
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

                if step % self.cfg['TRAIN']['SAVE_STEP'] == 0 or (step + 1) == self.cfg['TRAIN']['MAX_STEPS']:
                    checkpoint_path = os.path.join(FLAGS.whole_pretraining, 'model.ckpt')
                    saver.save(self.sess, checkpoint_path, global_step=step)

            coord.request_stop()
            coord.join(threads)
        print("done")

    '''
    Algorithm 1 in the paper
    '''
    def run_adversarial_training(self):
        if not os.path.exists(FLAGS.ckpt_dir):
            os.makedirs(FLAGS.ckpt_dir)

        use_pretrained_model = True
        graph = tf.Graph()
        with graph.as_default():
            self.global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
            videos_placeholder, utility_labels_placeholder, budget_labels_placeholder, dropout_placeholder, _ = \
                                            self.placeholder_inputs(self.cfg['TRAIN']['BATCH_SIZE'] * self.cfg['TRAIN']['GPU_NUM'])
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
                _lambda_op = tf.train.exponential_decay(FLAGS._lambda, global_step=self.global_step, decay_steps=10, decay_rate=0.9)
            else:
                _lambda_op = tf.identity(FLAGS._lambda)

            with tf.variable_scope(tf.get_variable_scope()):
                for gpu_index in range(0, self.cfg['TRAIN']['GPU_NUM']):
                    with tf.device('/gpu:%d' % gpu_index):
                        print('/gpu:%d' % gpu_index)
                        with tf.name_scope('%s_%d' % ('gpu', gpu_index)) as scope:
                            videos = videos_placeholder[gpu_index * self.cfg['TRAIN']['BATCH_SIZE']:(gpu_index + 1) * self.cfg['TRAIN']['BATCH_SIZE']]
                            utility_labels = utility_labels_placeholder[gpu_index * self.cfg['TRAIN']['BATCH_SIZE']:(gpu_index + 1) * self.cfg['TRAIN']['BATCH_SIZE']]
                            budget_labels = budget_labels_placeholder[gpu_index * self.cfg['TRAIN']['BATCH_SIZE']:(gpu_index + 1) * self.cfg['TRAIN']['BATCH_SIZE']]
                            loss_degrad, loss_budget, loss_utility, logits_budget, logits_utility, argmax_adverse_budget_loss = self.create_architecture_adversarial(scope, videos, utility_labels, budget_labels, dropout_placeholder, is_training_placeholder, self.cfg['TRAIN']['BATCH_SIZE'], _lambda_op)
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
            for multiplier in self.multiplier_lst:
                acc_op = accuracy(tf.concat(self.logits_budget_lst_dct[str(multiplier)], 0), budget_labels_placeholder)
                acc_op_lst.append(acc_op)
                loss_op = tf.reduce_max(self.loss_budget_lst_dct[str(multiplier)])
                loss_op_lst.append(loss_op)

            zero_ops_degrad, accum_ops_degrad, apply_gradient_op_degrad = self.create_grad_accum_for_late_update(opt_degrad, tower_grads_degrad, varlist_degrad, decay_with_global_step=True)
            zero_ops_budget, accum_ops_budget, apply_gradient_op_budget = self.create_grad_accum_for_late_update(opt_budget, tower_grads_budget, varlist_budget, decay_with_global_step=False)
            zero_ops_utility, accum_ops_utility, apply_gradient_op_utility = self.create_grad_accum_for_late_update(opt_utility, tower_grads_utility, varlist_utility+varlist_degrad, decay_with_global_step=False)

            tr_videos_op, tr_action_labels_op, tr_actor_labels_op = self.create_videos_reading_ops(is_train=True, is_val=False)
            val_videos_op, val_action_labels_op, val_actor_labels_op = self.create_videos_reading_ops(is_train=False, is_val=True)

            config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(graph=graph, config=config)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)
            init_op = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())
            self.sess.run(init_op)

            if use_pretrained_model:
                vardict_degradation = {v.name[:-2]: v for v in varlist_degrad}
                vardict_utility = {v.name[:-2]: v for v in varlist_utility}
                vardict = dict(vardict_degradation, **vardict_utility)
                restore_model_ckpt(self.sess, FLAGS.whole_pretraining, vardict, "DegradationModule")
            else:
                saver = tf.train.Saver(tf.trainable_variables())
                ckpt = tf.train.get_checkpoint_state(checkpoint_dir=FLAGS.ckpt_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(self.sess, ckpt.model_checkpoint_path)
                    print('Session restored from trained model at {}!'.format(ckpt.model_checkpoint_path))
                else:
                    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), FLAGS.ckpt_dir)

            saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=10)
            loss_summary_file, train_summary_file, test_summary_file, model_restarting_summary_file = self.create_summary_files()

            for step in xrange(self.cfg['TRAIN']['MAX_STEPS']):

                # Model initialization (only when step size=0) or model restarting
                if step == 0 or (FLAGS.use_restarting and step % FLAGS.restarting_step == 0):
                    budget_varlist = [v for v in tf.trainable_variables() if any(x in v.name for x in ["BudgetModule"])]
                    init_budget_op = tf.variables_initializer(budget_varlist)
                    self.sess.run(init_budget_op)
                    for _ in itertools.repeat(None, FLAGS.retraining_step):
                        start_time = time.time()
                        acc_util_lst, acc_budget_lst, loss_degrad_lst, loss_utility_lst, loss_budget_lst = [], [], [], [], []
                        self.sess.run(zero_ops_budget)
                        for _ in itertools.repeat(None, 20):
                            tr_videos, tr_action_labels, tr_actor_labels = self.sess.run(
                                [tr_videos_op, tr_action_labels_op, tr_actor_labels_op])
                            _, acc_util, acc_budget, loss_degrad_value, loss_utility_value, loss_budget_value = self.sess.run(
                                [accum_ops_budget, accuracy_util, accuracy_budget, loss_degrad_op,
                                 loss_utility_op, loss_budget_op],
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
                        self.sess.run(apply_gradient_op_budget)
                        assert not np.isnan(np.mean(loss_degrad_lst)), 'Model diverged with loss = NaN'
                        loss_summary = 'Restarting (Budget), Step: {:4d}, time: {:.4f}, budget loss: {:.8f}, training budget accuracy: {:.5f}, ' \
                                       'utility loss: {:.8f}, training utility accuracy: {:.5f}'.format(step,
                                        time.time() - start_time, np.mean(loss_budget_lst), np.mean(acc_budget_lst), np.mean(loss_utility_lst), np.mean(acc_util_lst))
                        model_restarting_summary_file.write(loss_summary + '\n')
                        print(loss_summary)

                # Part 1:
                # Train f_d using L_T+L_b' where L_b' is negative entropy or equivalently cross entropy with uniform labels.
                start_time = time.time()
                loss_degrad_lst, loss_utility_lst, loss_budget_lst = [], [], []
                self.sess.run(zero_ops_degrad)
                for _ in itertools.repeat(None, FLAGS.n_minibatches):
                    tr_videos, tr_actor_labels, tr_action_labels = self.sess.run(
                                    [tr_videos_op, tr_actor_labels_op, tr_action_labels_op])
                    _, argmax_cent, loss_degrad_value, loss_utility_value, loss_budget_value = self.sess.run([accum_ops_degrad,
                                                    argmax_adverse_budget_loss_op, loss_degrad_op, loss_utility_op, loss_budget_op],
                                                    feed_dict={videos_placeholder: tr_videos,
                                                               utility_labels_placeholder: tr_action_labels,
                                                               budget_labels_placeholder: tr_actor_labels,
                                                               dropout_placeholder: 1.0,
                                                               is_training_placeholder: True,
                                                               })
                    print(argmax_cent)
                    loss_degrad_lst.append(loss_degrad_value)
                    loss_utility_lst.append(loss_utility_value)
                    loss_budget_lst.append(loss_budget_value)
                _, _lambda = self.sess.run([apply_gradient_op_degrad, _lambda_op])

                assert not np.isnan(np.mean(loss_degrad_lst)), 'Model diverged with loss = NaN'
                loss_summary = 'Alternating Training (Degradation), Lambda: {:.8f}, Step: {:4d}, time: {:.4f}, degrad loss: {:.8f}, utility loss: {:.8f}, budget loss: {:.8f}'.format(_lambda, step,
                                        time.time() - start_time, np.mean(loss_degrad_lst), np.mean(loss_utility_lst), np.mean(loss_budget_lst))
                print(loss_summary)
                loss_summary_file.write(loss_summary + '\n')
                # End part 1

                # Part 2: End-to-end train Ft and Fd using L_T
                if FLAGS.use_monitor_utility:
                    while True:
                        start_time = time.time()
                        acc_util_lst, acc_budget_lst, loss_value_lst, loss_utility_lst, loss_budget_lst = [], [], [], [], []
                        for _ in itertools.repeat(None, FLAGS.n_minibatches):
                            val_videos, val_action_labels, val_actor_labels = self.sess.run([val_videos_op, val_action_labels_op, val_actor_labels_op])
                            acc_util, acc_budget, loss_value, loss_utility, loss_budget = self.sess.run(
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
                            # test_writer.add_summary(summary, step)
                        val_summary = "Step: {:4d}, time: {:.4f}, total loss: {:.8f}, utility loss: {:.8f}, budget loss: {:.8f}, validation utility accuracy: {:.5f}, validation budget accuracy: {:.5f},\n" .format(
                                step,
                                time.time() - start_time, np.mean(loss_value_lst),
                                np.mean(loss_utility_lst), np.mean(loss_budget_lst),
                                np.mean(acc_util_lst), np.mean(acc_budget_lst))
                        print(val_summary)

                        if np.mean(acc_util_lst) >= FLAGS.highest_util_acc_val:
                            break
                        start_time = time.time()
                        self.sess.run(zero_ops_utility)
                        acc_util_lst, acc_budget_lst, loss_degrad_lst, loss_utility_lst, loss_budget_lst = [], [], [], [], []
                        for _ in itertools.repeat(None, FLAGS.n_minibatches):
                            tr_videos, tr_action_labels, tr_actor_labels = self.sess.run([tr_videos_op, tr_action_labels_op, tr_actor_labels_op])
                            _, acc_util, acc_budget, loss_degrad_value, loss_utility_value, loss_budget_value = self.sess.run([accum_ops_utility, accuracy_util, accuracy_budget, loss_degrad_op, loss_utility_op, loss_budget_op],
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
                        self.sess.run([apply_gradient_op_utility])
                        assert not np.isnan(np.mean(loss_degrad_lst)), 'Model diverged with loss = NaN'
                        loss_summary = 'Alternating Training (Utility), Step: {:4d}, time: {:.4f}, degrad loss: {:.8f}, utility loss: {:.8f}, budget loss: {:.8f}, '.format(step,
                                                            time.time() - start_time, np.mean(loss_degrad_lst),
                                                            np.mean(loss_utility_lst), np.mean(loss_budget_lst))

                        print(loss_summary)
                # End part 2

                # Part 3: train Fb using L_b (cross entropy)
                if FLAGS.use_monitor_budget:
                    while True:
                        start_time = time.time()
                        self.sess.run(zero_ops_budget)
                        acc_util_lst, acc_budget_lst, loss_degrad_lst, loss_utility_lst, loss_budget_lst = [], [], [], [], []
                        for _ in itertools.repeat(None, FLAGS.n_minibatches):
                            tr_videos, tr_action_labels, tr_actor_labels = self.sess.run([tr_videos_op, tr_action_labels_op, tr_actor_labels_op])

                            _, acc_util, acc_budget, loss_degrad_value, loss_utility_value, loss_budget_value = self.sess.run([accum_ops_budget, accuracy_util, accuracy_budget, loss_degrad_op, loss_utility_op, loss_budget_op],
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
                        self.sess.run([apply_gradient_op_budget])
                        assert not np.isnan(np.mean(loss_degrad_lst)), 'Model diverged with loss = NaN'
                        loss_summary = 'Alternating Training (Budget), Step: {:4d}, time: {:.4f}, degrad loss: {:.8f}, utility loss: {:.8f}, budget loss: {:.8f}, ' \
                                   'training utility accuracy: {:.5f}, training budget accuracy: {:.5f}'.format(step,
                                                            time.time() - start_time, np.mean(loss_degrad_lst),
                                                            np.mean(loss_utility_lst), np.mean(loss_budget_lst),
                                                            np.mean(acc_util_lst), np.mean(acc_budget_lst))

                        print(loss_summary)
                        if np.mean(acc_budget_lst) >= FLAGS.highest_budget_acc_train:
                            break
                # End part 3

                if step % self.cfg['TRAIN']['VAL_STEP'] == 0:
                    start_time = time.time()
                    acc_util_train_lst, acc_budget_train_lst, loss_degrad_train_lst, loss_utility_train_lst, loss_budget_train_lst = [], [], [], [], []
                    acc_util_val_lst, acc_budget_val_lst, loss_degrad_val_lst, loss_utility_val_lst, loss_budget_val_lst = [], [], [], [], []
                    for _ in itertools.repeat(None, FLAGS.n_minibatches):
                        tr_videos, tr_action_labels, tr_actor_labels = self.sess.run(
                                    [tr_videos_op, tr_action_labels_op, tr_actor_labels_op])
                        acc_util, acc_budget, loss_degrad_value, loss_utility_value, loss_budget_value = self.sess.run([accuracy_util, accuracy_budget,
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

                    train_summary = "Step: {:4d}, time: {:.4f}, degrad loss: {:.8f}, utility loss: {:.8f}, budget loss: {:.8f}, training utility accuracy: {:.5f}, training budget accuracy: {:.5f}".format(
                            step,
                            time.time() - start_time, np.mean(loss_degrad_train_lst),
                            np.mean(loss_utility_train_lst), np.mean(loss_budget_train_lst),
                            np.mean(acc_util_train_lst), np.mean(acc_budget_train_lst))
                    print(train_summary)
                    train_summary_file.write(train_summary + '\n')

                    for _ in itertools.repeat(None, FLAGS.n_minibatches_eval):
                        val_videos, val_action_labels, val_actor_labels = self.sess.run(
                                        [val_videos_op, val_action_labels_op, val_actor_labels_op])
                        acc_util, acc_budget, loss_degrad_value, loss_utility_value, loss_budget_value = self.sess.run([accuracy_util, accuracy_budget,
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

                    test_summary = "Step: {:4d}, time: {:.4f}, degrad loss: {:.8f}, utility loss: {:.8f}, budget loss: {:.8f}, validation utility accuracy: {:.5f}, validation budget accuracy: {:.5f}".format(step,
                                                            time.time() - start_time, np.mean(loss_degrad_val_lst),
                                                            np.mean(loss_utility_val_lst), np.mean(loss_budget_val_lst),
                                                            np.mean(acc_util_val_lst), np.mean(acc_budget_val_lst))
                    print(test_summary)
                    test_summary_file.write(test_summary + '\n')

                if step % self.cfg['TRAIN']['SAVE_STEP'] == 0 or (step + 1) == self.cfg['TRAIN']['MAX_STEPS']:
                    checkpoint_path = os.path.join(FLAGS.ckpt_dir, 'model.ckpt')
                    saver.save(self.sess, checkpoint_path, global_step=step)

            loss_summary_file.close()
            train_summary_file.close()
            test_summary_file.close()
            coord.request_stop()
            coord.join(threads)
        print("done")

    '''
    Run testing of the trained model (direct test without any retraining, different from the two-fold-evaluation proposed in the paper)
    It will give the utility task accuracy and the privacy budget task accuracy
    '''
    def run_adversarial_testing(self):
        graph = tf.Graph()
        with graph.as_default():
            videos_placeholder, utility_labels_placeholder, budget_labels_placeholder, dropout_placeholder, _ = \
                                            self.placeholder_inputs(self.cfg['TEST']['BATCH_SIZE'] * self.cfg['TEST']['GPU_NUM'])
            is_training_placeholder = tf.placeholder(tf.bool)

            # Compute Acc
            logits_utility_lst, logits_budget_lst = [], []

            with tf.variable_scope(tf.get_variable_scope()) as scope:
                for gpu_index in range(0, self.cfg['TEST']['GPU_NUM']):
                    with tf.device('/gpu:%d' % gpu_index):
                        print('/gpu:%d' % gpu_index)
                        with tf.name_scope('%s_%d' % ('gpu', gpu_index)) as scope:
                            videos = videos_placeholder[gpu_index * self.cfg['TEST']['BATCH_SIZE']:(gpu_index + 1) * self.cfg['TEST']['BATCH_SIZE']]
                            utility_labels = utility_labels_placeholder[gpu_index * self.cfg['TEST']['BATCH_SIZE']:(gpu_index + 1) * self.cfg['TEST']['BATCH_SIZE']]
                            budget_labels = budget_labels_placeholder[gpu_index * self.cfg['TEST']['BATCH_SIZE']:(gpu_index + 1) * self.cfg['TEST']['BATCH_SIZE']]
                            _, _, _, logits_budget, logits_utility, _ = self.create_architecture_adversarial(scope, videos, utility_labels, budget_labels, dropout_placeholder, is_training_placeholder, self.cfg['TEST']['BATCH_SIZE'], None)
                            logits_budget_lst.append(logits_budget)
                            logits_utility_lst.append(logits_utility)
                            tf.get_variable_scope().reuse_variables()

            logits_utility = tf.concat(logits_utility_lst, 0)
            logits_budget = tf.concat(logits_budget_lst, 0)
            right_count_utility_op = tf.reduce_sum(
                tf.cast(tf.equal(tf.argmax(tf.nn.softmax(logits_utility), axis=1), utility_labels_placeholder),
                        tf.int32))
            # softmax_logits_utility_op = tf.nn.softmax(logits_utility)

            right_count_budget_op = tf.reduce_sum(
                tf.cast(tf.equal(tf.argmax(tf.nn.softmax(logits_budget), axis=1), budget_labels_placeholder),
                        tf.int32))
            # softmax_logits_budget_op = tf.nn.softmax(logits_budget)

            right_count_budget_op_lst = []
            for multiplier in self.multiplier_lst:
                logits = tf.concat(self.logits_budget_lst_dct['{}'.format(multiplier)], 0)
                right_count_op = tf.reduce_sum(
                    tf.cast(tf.equal(tf.argmax(tf.nn.softmax(logits), axis=1), budget_labels_placeholder),
                            tf.int32))
                right_count_budget_op_lst.append(right_count_op)

            videos_op, action_labels_op, actor_labels_op = self.create_videos_reading_ops(is_train=False, is_val=False)

            config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(graph=graph, config=config)

            init_op = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())
            self.sess.run(init_op)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)


            # Create a saver for writing training checkpoints.
            saver = tf.train.Saver(tf.trainable_variables())
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir=FLAGS.ckpt_dir)

            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(self.sess, ckpt.model_checkpoint_path)
                print('Session restored from trained model at {}!'.format(ckpt.model_checkpoint_path))
            else:
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), FLAGS.ckpt_dir)
            total_v_utility = 0.0
            total_v_budget = 0.0

            test_correct_num_utility = 0.0
            test_correct_num_budget = 0.0
            test_correct_num_budget_lst = [0.0] * FLAGS.NBudget

            try:
                while not coord.should_stop():
                    videos, utility_labels, budget_labels = self.sess.run([videos_op, action_labels_op, actor_labels_op])
                    feed = {videos_placeholder: videos, budget_labels_placeholder: budget_labels,
                            utility_labels_placeholder: utility_labels, is_training_placeholder: True,
                            dropout_placeholder: 1.0}
                    right_counts = self.sess.run(
                        [right_count_utility_op, right_count_budget_op] + right_count_budget_op_lst, feed_dict=feed)

                    test_correct_num_utility += right_counts[0]
                    total_v_utility += utility_labels.shape[0]

                    test_correct_num_budget += right_counts[1]
                    total_v_budget += budget_labels.shape[0]

                    for i in range(FLAGS.NBudget):
                        test_correct_num_budget_lst[i] += right_counts[i + 2]
                        # print(tf.argmax(softmax_logits, 1).eval(session=sess))
                        # print(logits.eval(feed_dict=feed, session=sess))
                        # print(labels)
            except tf.errors.OutOfRangeError:
                print('Done testing on all the examples')
            finally:
                coord.request_stop()

            with open('EvaluationResuls.txt', 'w') as wf:
                wf.write('Utility test acc: {},\ttest_correct_num: {},\ttotal_v: {}\n'.format(
                    test_correct_num_utility / total_v_utility, test_correct_num_utility, total_v_utility))
                wf.write('Budget ensemble test acc: {},\ttest_correct_num: {},\ttotal_v: {}\n'.format(
                    test_correct_num_budget / total_v_budget, test_correct_num_budget, total_v_budget))

                for i in range(FLAGS.NBudget):
                    wf.write('Budget{} test acc: {},\ttest_correct_num: {}\t: total_v: {}\n'.format(
                        self.multiplier_lst[i], test_correct_num_budget_lst[i] / total_v_budget,
                        test_correct_num_budget_lst[i], total_v_budget))

            coord.join(threads)
            self.sess.close()

def main(_):
    cfg = yaml.load(open('params.yml'))
    pp = pprint.PrettyPrinter()
    pp.pprint(FLAGS.__flags)
    pp.pprint(cfg)
    adversarial_training = AdversarialTraining(cfg)
    #adversarial_training.run_adversarial_training()
    #adversarial_training.run_pretraining_utility()
    #adversarial_training.run_pretraining_degrad()
    adversarial_training.run_adversarial_testing()

if __name__ == '__main__':
    tf.app.run()