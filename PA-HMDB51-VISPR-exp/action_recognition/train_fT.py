import sys, time, os, datetime, errno, pprint, itertools
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))

import tensorflow as tf
from common_flags import COMMON_FLAGS
from fT_flags import FLAGS
from modules.degradNet import fd
from modules.targetNet import fT
from loss import *
from utils import *
from nets import nets_factory

from sklearn.metrics import average_precision_score

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ["CUDA_VISIBLE_DEVICES"]=FLAGS.GPU_id


def placeholder_inputs(video_batch_size):
    videos_placeholder = tf.placeholder(tf.float32, shape=(video_batch_size, COMMON_FLAGS.DEPTH, COMMON_FLAGS.CROP_HEIGHT, COMMON_FLAGS.CROP_WIDTH, COMMON_FLAGS.NCHANNEL))
    labels_placeholder = tf.placeholder(tf.int64, shape=(video_batch_size))
    dropout_placeholder = tf.placeholder(tf.float32)
    return videos_placeholder, labels_placeholder, dropout_placeholder

def create_architecture(scope, videos, labels, dropout, factor):
    videos = bilinear_resize(videos, factor)
    logits = fT(videos, dropout)
    loss = tower_loss_xentropy_sparse(
        logits,
        labels,
        use_weight_decay=True,
        name_scope=scope,
    )

    return loss, logits

def get_varlists():
    varlist_fT = [v for v in tf.trainable_variables() if any(x in v.name for x in ["fT"])]
    varlist_fT_finetune = [v for v in varlist_fT if any(x in v.name.split('/')[1] for x in ["out", "d2"])]
    varlist_fT_main = list(set(varlist_fT) - set(varlist_fT_finetune))
    return varlist_fT, varlist_fT_finetune, varlist_fT_main

def build_graph(gpu_num, video_batch_size, factor=1, num_epochs=None):
    graph = tf.Graph()
    with graph.as_default():
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        videos_placeholder, fT_labels_placeholder, dropout_placeholder = placeholder_inputs(video_batch_size * gpu_num)


        tower_grads_fT_main, tower_grads_fT_finetune = [], []

        logits_lst = []
        loss_lst = []

        opt_fT_finetune = tf.train.AdamOptimizer(FLAGS.fT_finetune_lr)
        opt_fT_main = tf.train.AdamOptimizer(FLAGS.fT_main_lr)

        with tf.variable_scope(tf.get_variable_scope()) as scope:
            for gpu_index in range(0, gpu_num):
                with tf.device('/gpu:%d' % gpu_index):
                    print('/gpu:%d' % gpu_index)
                    with tf.name_scope('%s_%d' % ('gpu', gpu_index)) as scope:
                        videos = videos_placeholder[gpu_index * video_batch_size:(gpu_index + 1) * video_batch_size]
                        labels = fT_labels_placeholder[gpu_index * video_batch_size:(gpu_index + 1) * video_batch_size]

                        loss, logits = create_architecture(scope, videos, labels, dropout_placeholder, factor)

                        logits_lst.append(logits)
                        loss_lst.append(loss)

                        varlist_fT, varlist_fT_finetune, varlist_fT_main = get_varlists()
                        grads_fT_main = opt_fT_main.compute_gradients(loss, varlist_fT_main)
                        grads_fT_finetune = opt_fT_finetune.compute_gradients(loss, varlist_fT_finetune)

                        tower_grads_fT_main.append(grads_fT_main)
                        tower_grads_fT_finetune.append(grads_fT_finetune)

                        tf.get_variable_scope().reuse_variables()

        loss_fT_op = tf.reduce_mean(loss_lst, name='softmax')

        logits_fT_op = tf.concat(logits_lst, 0)
        acc_fT_op = accuracy(logits_fT_op, fT_labels_placeholder)

        right_count_fT_op = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(tf.nn.softmax(logits_fT_op), axis=1),
                                                                fT_labels_placeholder), tf.int32))

        zero_ops_fT_finetune, accum_ops_fT_finetune, apply_gradient_op_fT_finetune = create_grad_accum_for_late_update(opt_fT_finetune, tower_grads_fT_finetune, varlist_fT_finetune, FLAGS.n_minibatches, global_step, decay_with_global_step=False)
        zero_ops_fT_main, accum_ops_fT_main, apply_gradient_op_fT_main = create_grad_accum_for_late_update(opt_fT_main, tower_grads_fT_main, varlist_fT_main, FLAGS.n_minibatches, global_step, decay_with_global_step=False)

        tr_videos_op, tr_videos_labels_op = create_videos_reading_ops(is_train=True, is_val=False, GPU_NUM=gpu_num, BATCH_SIZE=video_batch_size, NUM_EPOCHS=num_epochs)
        val_videos_op, val_videos_labels_op = create_videos_reading_ops(is_train=False, is_val=True, GPU_NUM=gpu_num, BATCH_SIZE=video_batch_size, NUM_EPOCHS=num_epochs)
        test_videos_op, test_videos_labels_op = create_videos_reading_ops(is_train=False, is_val=False, GPU_NUM=gpu_num, BATCH_SIZE=video_batch_size, NUM_EPOCHS=num_epochs)

        init_op = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())

        return (graph, init_op,
                zero_ops_fT_finetune, accum_ops_fT_finetune, apply_gradient_op_fT_finetune,
                zero_ops_fT_main, accum_ops_fT_main, apply_gradient_op_fT_main,
                loss_fT_op, logits_fT_op, acc_fT_op, right_count_fT_op,
                tr_videos_op, tr_videos_labels_op,
                val_videos_op, val_videos_labels_op,
                test_videos_op, test_videos_labels_op,
                videos_placeholder, fT_labels_placeholder, dropout_placeholder,
                varlist_fT, varlist_fT_main, varlist_fT_finetune)

def update_fT(sess, step, n_minibatches,
                 zero_fT_finetune_op, apply_gradient_fT_finetune_op, accum_fT_finetune_op,
                 zero_fT_main_op, apply_gradient_fT_main_op, accum_fT_main_op,
                 loss_fT_op, videos_op, videos_labels_op,
                 videos_placeholder, fT_videos_labels_placeholder, dropout_placeholder):
    start_time = time.time()
    sess.run([zero_fT_finetune_op, zero_fT_main_op])
    loss_fT_lst = []
    for _ in itertools.repeat(None, n_minibatches):
        videos, videos_labels = sess.run(
            [videos_op, videos_labels_op])
        _, _, loss_fT = sess.run([accum_fT_finetune_op, accum_fT_main_op, loss_fT_op],
                                    feed_dict={videos_placeholder: videos,
                                               fT_videos_labels_placeholder: videos_labels,
                                               dropout_placeholder: 1.0,
                                               })
        loss_fT_lst.append(loss_fT)
    sess.run([apply_gradient_fT_finetune_op, apply_gradient_fT_main_op])
    loss_summary = 'Step: {:4d}, time: {:.4f}, fT loss: {:.8f}'.format(
        step,
        time.time() - start_time,
        np.mean(loss_fT_lst))
    return loss_summary

def eval_fT(sess, step, n_minibatches, loss_fT_op, acc_fT_op, videos_op, videos_labels_op,
            videos_placeholder, fT_videos_labels_placeholder, dropout_placeholder):
    start_time = time.time()
    acc_fT_lst, loss_fT_lst = [], []
    for _ in itertools.repeat(None, n_minibatches):
        videos, videos_labels = sess.run(
            [videos_op, videos_labels_op])
        acc_fT, loss_fT = sess.run(
            [acc_fT_op, loss_fT_op],
            feed_dict={videos_placeholder: videos,
                       fT_videos_labels_placeholder: videos_labels,
                       dropout_placeholder: 1.0})
        acc_fT_lst.append(acc_fT)
        loss_fT_lst.append(loss_fT)
    eval_summary = "Step: {:4d}, time: {:.4f}, fT loss: {:.8f}, fT accuracy: {:.5f},\n".format(
                    step, time.time() - start_time, np.mean(loss_fT_lst), np.mean(acc_fT_lst))
    return eval_summary

def train_fT(factor=1):
    # Create model directory
    ckpt_dir = os.path.join(FLAGS.ckpt_dir, str(factor))
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    (graph, init_op,
     zero_fT_finetune_op, accum_fT_finetune_op, apply_gradient_fT_finetune_op,
     zero_fT_main_op, accum_fT_main_op, apply_gradient_fT_main_op,
     loss_fT_op, logits_fT_op, acc_fT_op, right_count_fT_op,
     tr_videos_op, tr_videos_labels_op,
     val_videos_op, val_videos_labels_op,
     test_videos_op, test_videos_labels_op,
     videos_placeholder, fT_labels_placeholder, dropout_placeholder,
     varlist_fT, varlist_fT_main, varlist_fT_finetune) = build_graph(FLAGS.GPU_NUM, FLAGS.video_batch_size, factor=factor, num_epochs=None)


    use_pretrained_model = False

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True

    with tf.Session(graph=graph, config=config) as sess:

        #saver = tf.train.Saver(tf.trainable_variables() + bn_moving_vars)
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # Create a saver for writing training checkpoints.
        if use_pretrained_model:
            restore_model_pretrained_C3D(sess, COMMON_FLAGS.PRETRAINED_C3D_DIR, 'fT')
        else:
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir=ckpt_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Session restored from trained model at {}!'.format(ckpt.model_checkpoint_path))
            else:
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), ckpt_dir)

        saver = tf.train.Saver()
        for step in range(FLAGS.training_steps_fT):
            loss_summary = update_fT(sess, step, FLAGS.n_minibatches,
                                     zero_fT_finetune_op, apply_gradient_fT_finetune_op, accum_fT_finetune_op,
                                     zero_fT_main_op, apply_gradient_fT_main_op, accum_fT_main_op,
                                     loss_fT_op, tr_videos_op, tr_videos_labels_op,
                                     videos_placeholder, fT_labels_placeholder, dropout_placeholder)
            print("Updating fT, "+loss_summary)

            if step % FLAGS.val_step == 0:

                eval_summary = eval_fT(sess, step, 10, loss_fT_op, acc_fT_op, tr_videos_op, tr_videos_labels_op,
                                       videos_placeholder, fT_labels_placeholder, dropout_placeholder)
                print("TRAINING: "+eval_summary)

                eval_summary = eval_fT(sess, step, 10, loss_fT_op, acc_fT_op, val_videos_op, val_videos_labels_op,
                                       videos_placeholder, fT_labels_placeholder, dropout_placeholder)
                print("VALIDATION: "+eval_summary)

            if step % FLAGS.save_step == 0 or (step + 1) == FLAGS.training_steps_fT:
                checkpoint_path = os.path.join(ckpt_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

        coord.request_stop()
        coord.join(threads)

    print("done")

def test_fT(factor=1, is_training=False):

    ckpt_dir = os.path.join(FLAGS.ckpt_dir, str(factor))

    (graph, init_op,
     zero_ops_fT_finetune, accum_ops_fT_finetune, apply_gradient_op_fT_finetune,
     zero_ops_fT_main, accum_ops_fT_main, apply_gradient_op_fT_main,
     loss_fT_op, logits_fT_op, acc_fT_op, right_count_fT_op,
     tr_videos_op, tr_videos_labels_op,
     val_videos_op, val_videos_labels_op,
     test_videos_op, test_videos_labels_op,
     videos_placeholder, fT_labels_placeholder, dropout_placeholder,
     varlist_fT, varlist_fT_main, varlist_fT_finetune) = build_graph(FLAGS.GPU_NUM, FLAGS.video_batch_size, factor=factor, num_epochs=1)

    if is_training:
        videos_op, labels_op = tr_videos_op, tr_videos_labels_op
    else:
        videos_op, labels_op = test_videos_op, test_videos_labels_op,

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True
    with tf.Session(graph=graph, config=config) as sess:
        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # Create a saver for writing training checkpoints.
        saver = tf.train.Saver(varlist_fT)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir=ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Session restored from pretrained model at {}!'.format(ckpt.model_checkpoint_path))
        else:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), ckpt_dir)

        total_v_utility = 0.0
        test_correct_num_utility = 0.0
        try:
            while not coord.should_stop():
                videos, videos_labels = sess.run([videos_op, labels_op])
                feed = {videos_placeholder: videos, fT_labels_placeholder: videos_labels, dropout_placeholder: 1.0}
                right_utility = sess.run(right_count_fT_op, feed_dict=feed)
                print(total_v_utility)
                test_correct_num_utility += right_utility
                total_v_utility += videos_labels.shape[0]

        except tf.errors.OutOfRangeError:
            print('Done testing on all the examples')
        finally:
            coord.request_stop()
        coord.join(threads)

        save_dir = ckpt_dir
        isTraining = lambda bool: "training" if bool else "testing"
        with open(os.path.join(save_dir, 'fT_acc_{}.txt'.format(isTraining(is_training))), 'w') as wf:
            wf.write('fT test acc: {}'.format(test_correct_num_utility / total_v_utility))
            wf.write('fT test_correct_num: {}'.format(test_correct_num_utility))
            wf.write('fT total_v: {}'.format(total_v_utility))

def main(_):
    pp = pprint.PrettyPrinter()
    pp.pprint(FLAGS)
    #train_fT(factor=FLAGS.factor)
    test_fT(factor=FLAGS.factor, is_training=False)
    #test_fT(factor=FLAGS.factor, is_training=True)
    #for factor in [2, 4, 6, 8, 14, 16]:



if __name__ == '__main__':
    tf.app.run()
