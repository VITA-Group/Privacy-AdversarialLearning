'''
Two-Fold-Evaluation
First-fold: action (utility) prediction performance is preserved
Second-fold: privacy (budget) prediction performance is suppressed
'''
import sys, time, datetime, os, yaml, itertools
sys.path.insert(0, '..')

import numpy as np
import tensorflow as tf 

from input_data import *
from nets import nets_factory
from modules.degradlNet import residualNet
from loss import *
from utils import *
from validation import run_validation

from common_flags import COMMON_FLAGS
from twofold_flags import FLAGS

os.environ["CUDA_VISIBLE_DEVICES"]=FLAGS.GPU_id
print('Using GPU:', FLAGS.GPU_id)
print('GPU_NUM:', FLAGS.GPU_NUM)

cfg = yaml.load(open('params_2fold.yml'))

train_batch_size = cfg['TRAIN']['BATCH_SIZE'] * FLAGS.GPU_NUM 

model_max_steps_map = {
    'inception_v1': int(400*16*4/train_batch_size),
    'inception_v2': int(400*16*4/train_batch_size),
    'resnet_v1_50': int(400*16*4/train_batch_size),
    'resnet_v1_101': int(400*16*4/train_batch_size),
    'resnet_v2_50': int(400*16*4/train_batch_size),
    'resnet_v2_101': int(400*16*4/train_batch_size),
    'mobilenet_v1': int(400*16*4/train_batch_size), 
    'mobilenet_v1_075': int(1000*16*4/train_batch_size), 
    'mobilenet_v1_050': int(1000*16*4/train_batch_size),
    'mobilenet_v1_025': int(400*16*4/train_batch_size),
}
# Whether we need to train from scratch.
# Among the 10 evaluation models, 8 starts from imagenet pretrained model and 2 starts from scratch
model_train_from_scratch_map = {
    'inception_v1': False,
    'inception_v2': False,
    'resnet_v1_50': False,
    'resnet_v1_101': False,
    'resnet_v2_50': False,
    'resnet_v2_101': False,
    'mobilenet_v1': False,
    'mobilenet_v1_075': True,
    'mobilenet_v1_050': True,
    'mobilenet_v1_025': False,
}
# specific version of the models (useful only for mobilenetv1)
model_dir_map = {
    'inception_v1': 'inception_v1/inception_v1.ckpt',
    'inception_v2': 'inception_v2/inception_v2.ckpt',
    'resnet_v1_50': 'resnet_v1_50/resnet_v1_50.ckpt',
    'resnet_v1_101': 'resnet_v1_101/resnet_v1_101.ckpt',
    'resnet_v2_50': 'resnet_v2_50/resnet_v2_50.ckpt',
    'resnet_v2_101': 'resnet_v2_101/resnet_v2_101.ckpt',
    'mobilenet_v1': 'mobilenet_v1_1.0_128',
    'mobilenet_v1_075': 'mobilenet_v1_0.75_128',
    'mobilenet_v1_050': 'mobilenet_v1_0.50_128',
    'mobilenet_v1_025': 'mobilenet_v1_0.25_128',
}

model_name_lst = list(model_dir_map.keys())
model_name_lst = sorted(model_name_lst) 
print('model_name_lst:\n', model_name_lst)
'''
['inception_v1', 'inception_v2', 'mobilenet_v1', 'mobilenet_v1_025', 'mobilenet_v1_050', 'mobil
enet_v1_075', 'resnet_v1_101', 'resnet_v1_50', 'resnet_v2_101', 'resnet_v2_50']
'''

# build graph:
def build_graph(model_name):
    '''
    Returns:
        graph, init_op, train_op, 
        logits_op, acc_op, correct_count_op, loss_op, 
        tr_videos_op, tr_actor_labels_op, val_videos_op, val_actor_labels_op, test_videos_op, test_actor_labels_op,
        videos_placeholder, budget_labels_placeholder, 
        varlist_budget, varlist_degrad
    '''
    graph = tf.Graph()
    with graph.as_default():
        # global step:
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        # placholder inputs for graph:
        videos_placeholder, utility_labels_placeholder, budget_labels_placeholder, _, istraining_placeholder = \
            placeholder_inputs(train_batch_size)
        # degradation models:
        network_fn = nets_factory.get_network_fn(model_name,
                                                num_classes=COMMON_FLAGS.NUM_CLASSES_BUDGET,
                                                weight_decay=cfg['TRAIN']['WEIGHT_DECAY'],
                                                is_training=istraining_placeholder)
        # grads, logits, loss list:
        tower_grads = []
        logits_lst = []
        losses_lst = []
        # operation method:
        opt = tf.train.AdamOptimizer(1e-4)

        with tf.variable_scope(tf.get_variable_scope()) as scope:
            for gpu_index in range(0, FLAGS.GPU_NUM):
                with tf.device('/gpu:%d' % gpu_index):
                    print('/gpu:%d' % gpu_index)
                    with tf.name_scope('%s_%d' % ('gpu', gpu_index)) as scope:

                        videos = videos_placeholder[gpu_index * cfg['TRAIN']['BATCH_SIZE']:(gpu_index + 1) * cfg['TRAIN']['BATCH_SIZE']]
                        budget_labels = budget_labels_placeholder[gpu_index * cfg['TRAIN']['BATCH_SIZE']:(gpu_index + 1) * cfg['TRAIN']['BATCH_SIZE']]

                        degrad_videos = residualNet(videos, is_video=True)
                        # merge N and T channel in shape=(N,T,H,W,C)
                        degrad_videos = tf.reshape(degrad_videos, [cfg['TRAIN']['BATCH_SIZE'] * COMMON_FLAGS.DEPTH, COMMON_FLAGS.CROP_HEIGHT, COMMON_FLAGS.CROP_WIDTH, COMMON_FLAGS.NCHANNEL])
                        # logits:
                        logits, _ = network_fn(degrad_videos)
                        logits = tf.reshape(logits, [-1, COMMON_FLAGS.DEPTH, COMMON_FLAGS.NUM_CLASSES_BUDGET])
                        logits = tf.reduce_mean(logits, axis=1, keepdims=False)
                        # loss:
                        loss = tower_loss_xentropy_sparse(logits, budget_labels)
                        # append list:
                        logits_lst.append(logits)
                        losses_lst.append(loss)

                        # varible list of budget model:
                        varlist_budget = [v for v in tf.trainable_variables() if
                                            any(x in v.name for x in ["InceptionV1", "InceptionV2",
                                            "resnet_v1_50", "resnet_v1_101", "resnet_v2_50", "resnet_v2_101",
                                            "MobilenetV1_1.0", "MobilenetV1_0.75", "MobilenetV1_0.5", 'MobilenetV1_0.25', 
                                            'MobilenetV1'])]
                        # varible list of degrade model:
                        varlist_degrad = [v for v in tf.trainable_variables() if v not in varlist_budget]
                        # append grads:
                        tower_grads.append(opt.compute_gradients(loss, varlist_budget))

                        # reuse variables:
                        tf.get_variable_scope().reuse_variables()
        # loss tensor:
        loss_op = tf.reduce_mean(losses_lst)
        # acc tensor:
        logits_op = tf.concat(logits_lst, 0)
        acc_op = accuracy(logits_op, budget_labels_placeholder)
        # how many is correctly classified:
        correct_count_op = correct_num(logits_op, budget_labels_placeholder)
        # grads tensor:
        grads = average_gradients(tower_grads) # average gradient over all GPUs
        
        # apply gradients operation:
        with tf.control_dependencies([tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS))]):
            train_op = opt.apply_gradients(grads, global_step=global_step)

        # input operations:
        tr_videos_op, _, tr_actor_labels_op = create_videos_reading_ops(is_train=True, is_val=False, GPU_NUM=FLAGS.GPU_NUM, BATCH_SIZE=cfg['TRAIN']['BATCH_SIZE'])
        val_videos_op, _, val_actor_labels_op = create_videos_reading_ops(is_train=False, is_val=True, GPU_NUM=FLAGS.GPU_NUM, BATCH_SIZE=cfg['TRAIN']['BATCH_SIZE'])
        test_videos_op, _, test_actor_labels_op = create_videos_reading_ops(is_train=False, is_val=False, GPU_NUM=FLAGS.GPU_NUM, BATCH_SIZE=cfg['TEST']['BATCH_SIZE'])
        # initialize operations:
        init_op = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())

        return (graph, init_op, train_op, 
                logits_op, acc_op, correct_count_op, loss_op, 
                tr_videos_op, tr_actor_labels_op, val_videos_op, val_actor_labels_op, test_videos_op, test_actor_labels_op,
                videos_placeholder, utility_labels_placeholder, budget_labels_placeholder, istraining_placeholder,
                varlist_budget, varlist_degrad)

def run_training(start_from_trained_model, model_name):
    '''
    Args:
        model_name: name of the testing budget model.
    '''
    # Save ckpt of two-fold eval process in this directory:
    two_fold_eval_ckpt_dir = os.path.join(FLAGS.two_fold_eval_ckpt_dir, model_name)
    if not os.path.exists(two_fold_eval_ckpt_dir):
        os.makedirs(two_fold_eval_ckpt_dir)
    # Save summary files in this dir:
    summary_dir = os.path.join(FLAGS.summary_dir, model_name)
    if not os.path.exists(summary_dir):
        os.makedirs(summary_dir)
    train_summary_file = open(summary_dir + '/train_summary.txt', 'w')
    val_summary_file = open(summary_dir + '/val_summary.txt', 'w')

    # build graph:
    (graph, init_op, train_op, 
        _, acc_op, correct_count_op, loss_op, 
        tr_videos_op, tr_actor_labels_op, _, _, _, _,
        videos_placeholder, utility_labels_placeholder, budget_labels_placeholder, istraining_placeholder, 
        varlist_budget, varlist_degrad) = build_graph(model_name)
    
    # session configuration:
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    # config.gpu_options.allow_growth = True

    # run session:
    with tf.Session(graph=graph, config=config) as sess:
        '''
        In training, first run init_op, then do multi-threads.
        '''
        # initialize variables:
        sess.run(init_op)

        # multi threads:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        
        # Load ckpts:
        # for g in tf.global_variables():
        #     print(g.name)
        bn_moving_vars = [g for g in tf.global_variables() if 'moving_mean' in g.name]
        bn_moving_vars += [g for g in tf.global_variables() if 'moving_variance' in g.name]

        if start_from_trained_model:
            # load all parameters in the graph:
            restore_model_ckpt(sess=sess, ckpt_dir=two_fold_eval_ckpt_dir, varlist=tf.trainable_variables()+bn_moving_vars)
        else:
            # load degrade net:
            restore_model_ckpt(sess=sess, ckpt_dir=FLAGS.adversarial_ckpt_file_dir, varlist=varlist_degrad)
            # restore_model_ckpt(sess=sess, ckpt_dir=os.path.join(COMMON_FLAGS.pretrain_dir, 'degradation_models'), varlist=varlist_degrad)
            # load budget net:
            if not model_train_from_scratch_map[model_name]:
                pretrained_budget_model_ckpt_dir = os.path.join(
                    COMMON_FLAGS.hdd_dir, 'two_fold_evaluation', 'pretrained_budget_model', model_dir_map[model_name])
                varlist = [v for v in varlist_budget+bn_moving_vars if not any(x in v.name for x in ["logits"])]
                restore_model_ckpt(sess=sess, ckpt_dir=pretrained_budget_model_ckpt_dir, varlist=varlist)
        # End loading ckpts.

        # saver for saving all trainable variables (budget model+degrade model) ckpts:
        saver = tf.train.Saver(tf.trainable_variables() + bn_moving_vars, max_to_keep=1)
        
        best_val_acc = 0
        val_acc_lst = []
        for step in range(model_max_steps_map[model_name]):
            # updata on training data:
            start_time = time.time()
            train_videos, train_labels = sess.run([tr_videos_op, tr_actor_labels_op])
            _, acc, loss_value = sess.run([train_op, acc_op, loss_op], 
                feed_dict={videos_placeholder: train_videos, budget_labels_placeholder: train_labels, istraining_placeholder: True})
            assert not np.isnan(np.mean(loss_value)), 'Model diverged with loss = NaN'

            # print summary:
            if step % cfg['TRAIN']['PRINT_STEP'] == 0:
                summary = 'Step: {:4d}, time: {:.4f}, accuracy: {:.5f}, loss: {:.8f}'.format(step, time.time() - start_time, acc, np.mean(loss_value))
                print(summary)
                train_summary_file.write(summary + '\n')

            # validation on val set and save ckpt:
            if step % cfg['TRAIN']['VAL_STEP'] == 0 or (step + 1) == model_max_steps_map[model_name]:
                test_correct_num_lst, test_acc_lst, total_v = run_validation(sess=sess, 
					right_count_op_list=[correct_count_op],
					placeholder_list=[videos_placeholder, utility_labels_placeholder, budget_labels_placeholder, istraining_placeholder],
					batch_size=train_batch_size, dataset='val', istraining=True)
                # start summary:
                val_acc = test_acc_lst[0]
                summary = ("Step: %4d, validation accuracy: %.5f, total_v: %d") % (step, val_acc, total_v)
                print('Validation:\n' + summary)
                val_summary_file.write(summary + '\n')
                # end summary
                val_acc_lst.append(val_acc)
                # start saving model
                if val_acc > best_val_acc:
                    # update best_val_acc:
                    best_val_acc = val_acc
                    best_acc_step = step
                    print('Get new best val_acc: %f\n' % best_val_acc)
                    # Save checkpoint:
                    checkpoint_path = os.path.join(two_fold_eval_ckpt_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)
                    # end saving model
                # bn_temp_value = sess.run(bn_moving_vars[-1])
                # print('bn_temp_value:', bn_temp_value.shape, bn_temp_value[0:5])

        # join multi threads:
        coord.request_stop()
        coord.join(threads)

    print("done")
    np.save(os.path.join(summary_dir, 'val_acc_lst.npy'), np.array(val_acc_lst))

def run_testing(model_name):
    # save testing result in this dir:
    test_result_dir = FLAGS.test_result_dir.format(model_name)
    if not os.path.exists(test_result_dir):
        os.makedirs(test_result_dir)
    test_result_file = open(test_result_dir + '/test_result' + '.txt', 'a') # should be 'a' not 'w'.

    # build graph:
    (graph, init_op, _, 
    logits_op, _, correct_count_op, _, 
    _, _, val_videos_op, val_actor_labels_op, test_videos_op, test_actor_labels_op,
    videos_placeholder, utility_labels_placeholder, budget_labels_placeholder, istraining_placeholder, 
    varlist_budget, varlist_degrad) = build_graph(model_name)

    # session config:
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    # config.gpu_options.allow_growth = True

    # run session:
    with tf.Session(graph=graph, config=config) as sess:

        # load degrade and budget model ckpts:
        bn_moving_vars = [g for g in tf.global_variables() if 'moving_mean' in g.name]
        bn_moving_vars += [g for g in tf.global_variables() if 'moving_variance' in g.name]
        two_fold_eval_ckpt_dir = os.path.join(FLAGS.two_fold_eval_ckpt_dir, model_name)
        varlist = tf.trainable_variables()
        varlist += bn_moving_vars
        restore_model_ckpt(sess=sess, ckpt_dir=two_fold_eval_ckpt_dir, varlist=varlist)
        # end loading ckpts

        test_correct_num_lst, test_acc_lst, total_v = run_validation(sess=sess, 
					right_count_op_list=[correct_count_op],
					placeholder_list=[videos_placeholder, utility_labels_placeholder, budget_labels_placeholder, istraining_placeholder],
					batch_size=train_batch_size, dataset='test', istraining=True)
        
        # print and write testing result:
        test_result_str = model_name + '\n' + 'test acc: %.2f, test_correct_num:%d, total_v:%d\n' % (
            test_acc_lst[0], test_correct_num_lst[0], total_v)
        print(test_result_str)
        test_result_file.write(test_result_str)        

    print("done")


if __name__ == '__main__':
    start_from_trained_model = False
    # evaluate using N different budget models:
    for model_name in model_name_lst[3:]:   
        print('+++++++++++++++++++++++', model_name, '+++++++++++++++++++++++')
        # training:
        run_training(start_from_trained_model=start_from_trained_model, model_name = model_name)
        # testing:
        run_testing(model_name = model_name)