import sys, time, os, datetime, errno, pprint, itertools
from os.path import dirname, realpath
sys.path.append(dirname(dirname(dirname(realpath(__file__)))))

import tensorflow as tf
from common_flags import COMMON_FLAGS
from adversarial_training.pre_training.pretrain_flags import FLAGS
from modules.degradNet import fd
from modules.targetNet import fT
from loss import *
from utils import *
from nets import nets_factory

from sklearn.metrics import average_precision_score

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ["CUDA_VISIBLE_DEVICES"]=FLAGS.GPU_id


def placeholder_inputs(image_batch_size):
    images_placeholder = tf.placeholder(tf.float32, shape=(image_batch_size, COMMON_FLAGS.CROP_HEIGHT, COMMON_FLAGS.CROP_WIDTH, COMMON_FLAGS.NCHANNEL))
    budget_labels_placeholder = tf.placeholder(tf.float32, shape=(image_batch_size, COMMON_FLAGS.NUM_CLASSES_BUDGET))
    isTraining_placeholder = tf.placeholder(tf.bool)
    return images_placeholder, budget_labels_placeholder, isTraining_placeholder

def create_architecture(scope, loss_fb_lst_dict, logits_fb_lst_dict, fb_dict, 
    image_batch_size, images, fb_labels):
    '''
    Create the network structure: images -> fd -> fb -> fb_labels

    Args:
        fb_dict: dictionary. {model_name (str): model_structure (tf model)}
        images: frames of input videos.

    Returns:
        loss_fb_images_op: tf.Operation. Opteration for finding budget task loss.
        logits_fb_images_op: tf.Operation. Opteration for finding budget task forward logits.
    '''
    
    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
        fd_images = fd(images)
    loss_fb_images = 0.0
    logits_fb_images = tf.zeros([image_batch_size, COMMON_FLAGS.NUM_CLASSES_BUDGET])
    for name, fb in fb_dict.items():
        print(name)
        with tf.variable_scope(tf.get_variable_scope(), reuse=False):
            logits, _ = fb(fd_images)
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                                    labels=fb_labels))
        logits_fb_images += logits
        logits_fb_lst_dict[name].append(logits)
        loss_fb_lst_dict[name].append(loss)
        loss_fb_images += loss
    loss_fb_images_op = tf.divide(loss_fb_images, len(fb_dict.items()), 'LossFbMean')
    logits_fb_images_op = tf.divide(logits_fb_images, len(fb_dict.items()), 'LogitsFbMean')

    return loss_fb_images_op, logits_fb_images_op

def get_varlists():
    varlist_fd = [v for v in tf.trainable_variables() if any(x in v.name for x in ["fd"])]
    varlist_fb = [v for v in tf.trainable_variables() if
                      not any(x in v.name for x in ["fT", "fd"])]
    return varlist_fd, varlist_fb

def build_graph(gpu_num, image_batch_size):
    from collections import defaultdict
    logits_fb_lst_dict = defaultdict(list)
    loss_fb_lst_dict = defaultdict(list)
    graph = tf.Graph()
    with graph.as_default():
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        images_placeholder, fb_labels_placeholder, isTraining_placeholder = placeholder_inputs(image_batch_size * gpu_num)

        tower_grads_fb = []
        logits_fb_images_lst = []
        loss_fb_images_lst = []

        opt_fb = tf.train.AdamOptimizer(FLAGS.fb_lr)

        fb_name_lst = ['resnet_v1_50', 'resnet_v2_50', 'mobilenet_v1', 'mobilenet_v1_075']
        fb_dict = {}
        for model_name in fb_name_lst:
            fb_dict[model_name] = nets_factory.get_network_fn(
                model_name,
                num_classes=COMMON_FLAGS.NUM_CLASSES_BUDGET,
                weight_decay=FLAGS.weight_decay,
                is_training=isTraining_placeholder)
        with tf.variable_scope(tf.get_variable_scope()) as scope:
            for gpu_index in range(0, gpu_num):
                with tf.device('/gpu:%d' % gpu_index):
                    print('/gpu:%d' % gpu_index)
                    with tf.name_scope('%s_%d' % ('gpu', gpu_index)) as scope:
                        images = images_placeholder[gpu_index * image_batch_size:(gpu_index + 1) * image_batch_size]
                        fb_labels = fb_labels_placeholder[gpu_index * image_batch_size:(gpu_index + 1) * image_batch_size, :]

                        loss_fb_images, logits_fb_images = create_architecture(
                            scope, loss_fb_lst_dict, logits_fb_lst_dict, fb_dict, image_batch_size, images, fb_labels)

                        logits_fb_images_lst.append(logits_fb_images)
                        loss_fb_images_lst.append(loss_fb_images)

                        varlist_fb, varlist_fd = get_varlists()
                        grads_fb = opt_fb.compute_gradients(loss_fb_images, varlist_fb)

                        tower_grads_fb.append(grads_fb)

                        tf.get_variable_scope().reuse_variables()

        loss_fb_op = tf.reduce_mean(loss_fb_images_lst, name='softmax')
        logits_fb_op = tf.concat(logits_fb_images_lst, 0)
        zero_ops_fb, accum_ops_fb, apply_gradient_op_fb = create_grad_accum_for_late_update(opt_fb, tower_grads_fb, varlist_fb, FLAGS.n_minibatches, global_step, decay_with_global_step=False)

        tr_images_op, tr_images_labels_op = create_images_reading_ops(is_train=True, is_val=False,
                                                                      GPU_NUM=gpu_num, BATCH_SIZE=image_batch_size)
        val_images_op, val_images_labels_op = create_images_reading_ops(is_train=False, is_val=True,
                                                                    GPU_NUM=gpu_num, BATCH_SIZE=image_batch_size)
        init_op = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())

        varlist_bn = [g for g in tf.global_variables() if 'moving_mean' in g.name]
        varlist_bn += [g for g in tf.global_variables() if 'moving_variance' in g.name]

        return (graph, init_op,
                zero_ops_fb, accum_ops_fb, apply_gradient_op_fb,
                loss_fb_op, logits_fb_op,
                tr_images_op, tr_images_labels_op,
                val_images_op, val_images_labels_op,
                images_placeholder, fb_labels_placeholder, isTraining_placeholder,
                varlist_fb, varlist_fd, varlist_bn)

def update_fb(sess, step, n_minibatches, zero_fb_op, apply_gradient_fb_op, accum_fb_op, loss_fb_op,
              images_op, images_labels_op, 
              images_placeholder, fb_images_labels_placeholder, isTraining_placeholder):
    '''
    pretrain fb with fixed fd.
    '''
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
    loss_summary = 'Step: {:4d}, time: {:.4f}, budget loss: {:.8f}'.format(
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

def run_pretraining_fb():
    '''
    Pretrain fb. fb is fixed as the pretrained value abtained by run_pretraining_fT() function.
    '''
    # Create model directory
    if not os.path.exists(FLAGS.pretrained_fb_ckpt_dir):
        os.makedirs(FLAGS.pretrained_fb_ckpt_dir)

    (graph, init_op,
    zero_ops_fb, accum_ops_fb, apply_gradient_op_fb,
    loss_fb_op, logits_fb_op,
    tr_images_op, tr_images_labels_op,
    val_images_op, val_images_labels_op,
    images_placeholder, fb_labels_placeholder, isTraining_placeholder,
    varlist_fb, varlist_fd, varlist_bn) = build_graph(FLAGS.GPU_NUM, FLAGS.image_batch_size)


    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True

    with tf.Session(graph=graph, config=config) as sess:

        #saver = tf.train.Saver(tf.trainable_variables() + bn_moving_vars)
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # Create a saver for writing training checkpoints.
        saver = tf.train.Saver()
    
        # load pretrained fd:
        restore_model_ckpt(sess, FLAGS.pretrained_fT_ckpt_dir, varlist_fd, "fd")

        for step in range(FLAGS.pretraining_steps_fb):

            loss_summary = update_fb(sess, step, FLAGS.n_minibatches, zero_ops_fb, apply_gradient_op_fb, accum_ops_fb, loss_fb_op,
                      tr_images_op, tr_images_labels_op, images_placeholder, fb_labels_placeholder, isTraining_placeholder)
            print("Updating fb, " + loss_summary)

            if step % FLAGS.val_step == 0:
                eval_summary = eval_fb(sess, step, 30, logits_fb_op, loss_fb_op, tr_images_op, tr_images_labels_op,
                                       images_placeholder, fb_labels_placeholder, isTraining_placeholder)
                print("TRAINING: " + eval_summary)
                eval_summary = eval_fb(sess, step, 30, logits_fb_op, loss_fb_op, val_images_op, val_images_labels_op,
                                       images_placeholder, fb_labels_placeholder,
                                       isTraining_placeholder)
                print("VALIDATION: " + eval_summary)

            if step % FLAGS.save_step == 0 or (step + 1) == FLAGS.pretraining_steps_fbfdfT:
                checkpoint_path = os.path.join(FLAGS.pretrained_fb_ckpt_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    run_pretraining_fb()
