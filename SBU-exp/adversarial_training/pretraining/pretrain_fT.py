import sys, time, os, itertools
sys.path.insert(0, '..')

import input_data 
from modules.degradlNet import residualNet
from modules.utilityNet import utilityNet
from loss import *
from utils import *
from validation import run_validation

from common_flags import COMMON_FLAGS
from pretrain_flags import FLAGS

os.environ["CUDA_VISIBLE_DEVICES"]=FLAGS.GPU_id
print('Using GPU:', FLAGS.GPU_id)
print('GPU_NUM:', FLAGS.GPU_NUM)

MAX_STEPS = 500
VAL_STEP = 20
SAVE_STEP = 50
TRAIN_BATCH_SIZE = 2

def create_architecture_pretraining_fT(scope, videos, utility_labels, dropout_placeholder):
    # fd part:
    degrad_videos = residualNet(videos, is_video=True)
    degrad_videos = avg_replicate(degrad_videos) if FLAGS.use_avg_replicate else degrad_videos
    # fd part ends.
    # fT part:
    logits_utility = utilityNet(degrad_videos, dropout_placeholder, wd=0.001)
    loss_utility = tower_loss_xentropy_sparse(logits_utility, utility_labels, use_weight_decay=True, name_scope=scope)
    # fT part ends.
    return loss_utility, logits_utility

def run_pretraining_fT(start_from_trained_model):
    '''
    Initialize f_T on pretrained f_d 
    Args:
        start_from_trained_model: boolean. If False, use sports1M initialized fT. If true, use pretrained fT.
    '''
    degradation_ckpt_dir = os.path.join(COMMON_FLAGS.pretrain_dir, 'degradation_models')
    target_ckpt_dir = os.path.join(COMMON_FLAGS.pretrain_dir, 'target_models')
    if not os.path.exists(target_ckpt_dir):
        os.makedirs(target_ckpt_dir)

    # define graph:
    graph = tf.Graph()
    with graph.as_default():
        # global step:
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        # placeholder inputs:
        videos_placeholder, utility_labels_placeholder, budget_labels_placeholder, dropout_placeholder, istraining_placeholder = \
                                        placeholder_inputs(TRAIN_BATCH_SIZE * FLAGS.GPU_NUM)
        tower_grads_degrad, tower_grads_utility = [], []

        # Compute Acc
        logits_utility_lst = []

        # Compute Loss
        loss_utility_lst = []

        # optimizations:
        opt_degrad = tf.train.AdamOptimizer(1e-3)
        opt_utility = tf.train.AdamOptimizer(1e-5)

        with tf.variable_scope(tf.get_variable_scope()):
            for gpu_index in range(0, FLAGS.GPU_NUM):
                with tf.device('/gpu:%d' % gpu_index):
                    print('/gpu:%d' % gpu_index)
                    with tf.name_scope('%s_%d' % ('gpu', gpu_index)) as scope:
                        videos = videos_placeholder[gpu_index * TRAIN_BATCH_SIZE : (gpu_index + 1) * TRAIN_BATCH_SIZE]
                        utility_labels = utility_labels_placeholder[gpu_index * TRAIN_BATCH_SIZE : (gpu_index + 1) * TRAIN_BATCH_SIZE]

                        loss_utility, logits_utility = create_architecture_pretraining_fT(scope, videos, utility_labels, dropout_placeholder)
                        logits_utility_lst.append(logits_utility)
                        loss_utility_lst.append(loss_utility)

                        varlist_degrad = [v for v in tf.trainable_variables() if
                                                        any(x in v.name for x in ["DegradationModule"])]
                        varlist_utility = [v for v in tf.trainable_variables() if
                                                        any(x in v.name for x in ["UtilityModule"])]

                        grads_degrad = opt_degrad.compute_gradients(loss_utility, varlist_degrad)
                        grads_utility = opt_utility.compute_gradients(loss_utility, varlist_utility)

                        tower_grads_degrad.append(grads_degrad)
                        tower_grads_utility.append(grads_utility)

                        # Reuse variables for the next tower.
                        tf.get_variable_scope().reuse_variables()

        loss_utility_op = tf.reduce_mean(loss_utility_lst, name='softmax')
        logits_utility = tf.concat(logits_utility_lst, 0)
        right_count_utility_op = correct_num(logits_utility, utility_labels_placeholder)

        zero_ops_degrad, accum_ops_degrad, apply_gradient_op_degrad = create_grad_accum_for_late_update(
            opt_degrad, tower_grads_degrad, varlist_degrad, FLAGS.n_minibatches, global_step, decay_with_global_step=False)
        zero_ops_utility, accum_ops_utility, apply_gradient_op_utility = create_grad_accum_for_late_update(
            opt_utility, tower_grads_utility, varlist_utility, FLAGS.n_minibatches, global_step, decay_with_global_step=False)

        tr_videos_op, tr_action_labels_op, _ = create_videos_reading_ops(is_train=True, is_val=False, GPU_NUM=FLAGS.GPU_NUM, BATCH_SIZE=TRAIN_BATCH_SIZE)
        # val_videos_op, val_action_labels_op, _ = create_videos_reading_ops(is_train=False, is_val=True, GPU_NUM=FLAGS.GPU_NUM, BATCH_SIZE=TRAIN_BATCH_SIZE)

        init_op = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())

    # session config:
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True
    # session:
    with tf.Session(graph=graph, config=config) as sess:
        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # Create a saver for writing training checkpoints.
        restore_model_ckpt(sess=sess, ckpt_dir=degradation_ckpt_dir, varlist=varlist_degrad)
        if start_from_trained_model:
            restore_model_ckpt(sess=sess, ckpt_dir=target_ckpt_dir, varlist=varlist_utility)
        else:
            # variable name and output tensor shape of the last layer are different between ckpt and our graph, so we must convert:
            temp_varlist = [v for v in varlist_utility if not any(x in v.name.split('/')[1] for x in ["out", "d2"])]
            temp_vardict = {v.name[:-2].replace('UtilityModule', 'var_name'): v for v in temp_varlist}
            restore_model_ckpt(sess=sess, ckpt_dir=COMMON_FLAGS.PRETRAINED_C3D, varlist=temp_vardict)

        # saver:
        saver = tf.train.Saver(varlist_utility, max_to_keep=1)
        save_checkpoint_path = os.path.join(target_ckpt_dir, 'model.ckpt')

        # train:
        for step in range(MAX_STEPS):
            start_time = time.time()
            sess.run([zero_ops_utility, zero_ops_degrad])
            loss_utility_lst = []
            for _ in itertools.repeat(None, FLAGS.n_minibatches):
                tr_videos, tr_videos_labels = sess.run([tr_videos_op, tr_action_labels_op])
                _, loss_utility = sess.run([accum_ops_utility, loss_utility_op],
                                    feed_dict={videos_placeholder: tr_videos, utility_labels_placeholder: tr_videos_labels, dropout_placeholder: 1.0})
                loss_utility_lst.append(loss_utility)
            sess.run([apply_gradient_op_utility, apply_gradient_op_degrad])
            loss_summary = 'Utility Module + Degradation Module, Step: {:4d}, time: {:.4f}, utility loss: {:.8f}'.format(
                    step, time.time() - start_time, np.mean(loss_utility_lst))
            print(loss_summary)

            # validation on utility task:
            if step % VAL_STEP == 0:
                start_time = time.time()
                test_correct_num_lst, test_acc_lst, total_v = run_validation(sess=sess, 
                    right_count_op_list=[right_count_utility_op], 
                    placeholder_list=[videos_placeholder, utility_labels_placeholder, budget_labels_placeholder, dropout_placeholder, istraining_placeholder], 
                    batch_size=TRAIN_BATCH_SIZE*FLAGS.GPU_NUM, dataset='val')

                test_summary = "Step: {:4d}, time: {:.4f}, validation utility correct num: {:.8f}, accuracy: {:.5f}".format(
                        step, time.time() - start_time, test_correct_num_lst[0], test_acc_lst[0])
                print(test_summary)

            # save model:
            if step % SAVE_STEP == 0 or (step + 1) == MAX_STEPS:
                saver.save(sess, save_checkpoint_path, global_step=step)

        coord.request_stop()
        coord.join(threads)
    print("done")

if __name__ == '__main__':
    start_from_trained_model = False
    run_pretraining_fT(start_from_trained_model)