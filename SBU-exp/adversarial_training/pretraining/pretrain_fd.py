'''
Pretrain fd as close to identity mapping. 
'''
import sys, time, errno, os
sys.path.insert(0, '..')

import input_data
from modules.degradlNet import residualNet
from loss import *
from utils import create_images_reading_ops, restore_model_ckpt

from common_flags import COMMON_FLAGS
from pretrain_flags import FLAGS

import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL']=2
os.environ["CUDA_VISIBLE_DEVICES"]="1"

# config:
MAX_STEPS = 16730
VAL_STEP = 50
SAVE_STEP = -1
TRAIN_BATCH_SIZE = 32


def create_architecture_fd(scope, images, labels):
    degrad_images = residualNet(images, is_video=False)
    loss = tower_loss_mse(scope, degrad_images, labels)
    return loss

def run_pretraining_fd(start_from_trained_model):
    '''
    Pretrain f_d to be identity mapping of the input
    Args:
        start_from_trained_model: boolean. If False, use random initialized fd. If true, use pretrained fd.
    '''
    degradation_ckpt_dir = os.path.join(COMMON_FLAGS.pretrain_dir, 'degradation_models')
    if not os.path.exists(degradation_ckpt_dir):
        os.makedirs(degradation_ckpt_dir)
    log_dir = os.path.join(degradation_ckpt_dir, 'tfsummary')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    graph = tf.Graph()
    with graph.as_default():
        batch_size = TRAIN_BATCH_SIZE * FLAGS.GPU_NUM
        images_placeholder = tf.placeholder(tf.float32, [batch_size, 120, 160, 3], name='images')
        labels_placeholder = tf.placeholder(tf.float32, [batch_size, 120, 160, 3], name='labels')
        tower_grads = []
        losses = []
        opt = tf.train.AdamOptimizer(1e-3)

        with tf.variable_scope(tf.get_variable_scope()) as scope:
            for gpu_index in range(0,FLAGS.GPU_NUM):
                with tf.device('/gpu:%d' % gpu_index):
                    print('/gpu:%d' % gpu_index)
                    with tf.name_scope('%s_%d' % ('gpu', gpu_index)) as scope:
                        images = images_placeholder[gpu_index * TRAIN_BATCH_SIZE:(gpu_index + 1) * TRAIN_BATCH_SIZE]
                        labels = labels_placeholder[gpu_index * TRAIN_BATCH_SIZE:(gpu_index + 1) * TRAIN_BATCH_SIZE]
                        loss = create_architecture_fd(scope, images, labels)
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

        tr_images_op, tr_labels_op = create_images_reading_ops(is_train=True, is_val=False, GPU_NUM=FLAGS.GPU_NUM, BATCH_SIZE=TRAIN_BATCH_SIZE)
        val_images_op, val_labels_op = create_images_reading_ops(is_train=False, is_val=True, GPU_NUM=FLAGS.GPU_NUM, BATCH_SIZE=TRAIN_BATCH_SIZE)

    conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    conf.gpu_options.allow_growth = True

    with tf.Session(graph=graph, config=conf) as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        init_op = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())
        sess.run(init_op)

        if start_from_trained_model:
            restore_model_ckpt(sess=sess, ckpt_dir=degradation_ckpt_dir, varlist=tf.trainable_variables())

        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
        val_writer = tf.summary.FileWriter(log_dir + '/val', sess.graph)

        # saver:
        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=1)
        save_checkpoint_path = os.path.join(degradation_ckpt_dir, 'model.ckpt')

        # train:
        print("Training...")
        for step in range(MAX_STEPS):
            # Run by batch images
            start_time = time.time()
            tr_images, tr_labels = sess.run([tr_images_op, tr_labels_op])
            print(tr_images.shape)
            print(tr_labels.shape)
            tr_feed = {images_placeholder: tr_images, labels_placeholder: tr_labels}
            _, loss_value = sess.run([train_op, loss_op], feed_dict=tr_feed)
            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
            print("Step: [%2d], time: [%4.4f], training_loss = [%.8f]" % (step, time.time() - start_time, loss_value))
            if step % VAL_STEP == 0:
                val_images, val_labels = sess.run([val_images_op, val_labels_op])
                val_feed = {images_placeholder: val_images, labels_placeholder: val_labels}
                summary, loss_value, psnr = sess.run([merged, loss_op, psnr_op], feed_dict=val_feed)
                print("Step: [%2d], time: [%4.4f], validation_loss = [%.8f], validation_psnr = [%.8f]" %
                        (step, time.time() - start_time, loss_value, psnr))
                val_writer.add_summary(summary, step)
                tr_images, tr_labels = sess.run([tr_images_op, tr_labels_op])
                tr_feed = {images_placeholder: tr_images, labels_placeholder: tr_labels}
                summary, loss_value, psnr = sess.run([merged, loss_op, psnr_op], feed_dict=tr_feed)
                print("Step: [%2d], time: [%4.4f], training_loss = [%.8f], training_psnr = [%.8f]" %
                        (step, time.time() - start_time, loss_value, psnr))
                train_writer.add_summary(summary, step)
            if (step+1) % SAVE_STEP == 0 or (step + 1) == MAX_STEPS:
                saver.save(sess, save_checkpoint_path, global_step=step)

        coord.request_stop()
        coord.join(threads)
        sess.close()

if __name__=='__main__':
    start_from_trained_model = True
    run_pretraining_fd(start_from_trained_model)