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


def placeholder_inputs(video_batch_size, image_batch_size):
    videos_placeholder = tf.placeholder(tf.float32, shape=(video_batch_size, COMMON_FLAGS.DEPTH, COMMON_FLAGS.CROP_HEIGHT, COMMON_FLAGS.CROP_WIDTH, COMMON_FLAGS.NCHANNEL))
    images_placeholder = tf.placeholder(tf.float32, shape=(image_batch_size, COMMON_FLAGS.CROP_HEIGHT, COMMON_FLAGS.CROP_WIDTH, COMMON_FLAGS.NCHANNEL))
    utility_labels_placeholder = tf.placeholder(tf.int64, shape=(video_batch_size))
    budget_labels_placeholder = tf.placeholder(tf.float32, shape=(image_batch_size, COMMON_FLAGS.NUM_CLASSES_BUDGET))
    dropout_placeholder = tf.placeholder(tf.float32)
    isTraining_placeholder = tf.placeholder(tf.bool)
    return videos_placeholder, images_placeholder, utility_labels_placeholder, \
           budget_labels_placeholder, dropout_placeholder, isTraining_placeholder

def create_architecture(scope, loss_fb_lst_dct, logits_fb_lst_dct, fb_dict, video_batch_size, image_batch_size, videos, images, fT_labels, fb_labels, dropout):
    videos = tf.reshape(videos, [video_batch_size * COMMON_FLAGS.DEPTH, COMMON_FLAGS.CROP_HEIGHT, COMMON_FLAGS.CROP_WIDTH, COMMON_FLAGS.NCHANNEL])
    videos = fd(videos)
    videos_utility = tf.reshape(videos,
                                [video_batch_size, COMMON_FLAGS.DEPTH, COMMON_FLAGS.CROP_HEIGHT, COMMON_FLAGS.CROP_WIDTH, COMMON_FLAGS.NCHANNEL])
    logits_utility = fT(videos_utility, dropout)
    loss_utility = tower_loss_xentropy_sparse(
        logits_utility,
        fT_labels,
        use_weight_decay=True,
        name_scope=scope,
    )
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
        logits_fb_lst_dct[name].append(logits)
        loss_fb_lst_dct[name].append(loss)
        loss_fb_images += loss
    loss_fb_images_op = tf.divide(loss_fb_images, 4.0, 'LossFbMean')
    logits_fb_images_op = tf.divide(logits_fb_images, 4.0, 'LogitsFbMean')

    return loss_utility, logits_utility, loss_fb_images_op, logits_fb_images_op

def get_varlists():
    varlist_fd = [v for v in tf.trainable_variables() if any(x in v.name for x in ["fd"])]
    varlist_fT = [v for v in tf.trainable_variables() if any(x in v.name for x in ["fT"])]
    varlist_fT_finetune = [v for v in varlist_fT if any(x in v.name.split('/')[1] for x in ["out", "d2"])]
    varlist_fb = [v for v in tf.trainable_variables() if
                      not any(x in v.name for x in ["fT", "fd"])]
    varlist_fT_main = list(set(varlist_fT) - set(varlist_fT_finetune))
    return varlist_fd, varlist_fT, varlist_fT_finetune, varlist_fT_main, varlist_fb

def build_graph(gpu_num, video_batch_size, image_batch_size):
    from collections import defaultdict
    logits_fb_lst_dct = defaultdict(list)
    loss_fb_lst_dct = defaultdict(list)
    graph = tf.Graph()
    with graph.as_default():
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        videos_placeholder, images_placeholder, fT_labels_placeholder, \
        fb_labels_placeholder, dropout_placeholder, isTraining_placeholder = placeholder_inputs(
                            video_batch_size * gpu_num, image_batch_size * gpu_num)


        tower_grads_fd, tower_grads_fT_main, tower_grads_fT_finetune, tower_grads_fb = [], [], [], []

        logits_fT_videos_lst = []
        logits_fb_images_lst = []

        loss_fT_videos_lst = []
        loss_fb_images_lst = []

        opt_fd = tf.train.AdamOptimizer(FLAGS.fd_lr)
        opt_fT_finetune = tf.train.AdamOptimizer(FLAGS.fT_finetune_lr)
        opt_fT_main = tf.train.AdamOptimizer(FLAGS.fT_main_lr)
        opt_fb = tf.train.AdamOptimizer(FLAGS.fb_lr)

        fb_name_lst = ['resnet_v1_50', 'resnet_v2_50', 'mobilenet_v1', 'mobilenet_v1_075']
        # budgetNet_model_name_lst = ['resnet_v1_50', 'resnet_v2_50', 'mobilenet_v1']
        # budgetNet_model_name_lst = ['resnet_v1_50', 'resnet_v2_50']
        # budgetNet_model_name_lst = ['resnet_v1_50']
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
                        videos = videos_placeholder[gpu_index * video_batch_size:(gpu_index + 1) * video_batch_size]
                        images = images_placeholder[gpu_index * image_batch_size:(gpu_index + 1) * image_batch_size]
                        fT_labels = fT_labels_placeholder[gpu_index * video_batch_size:(gpu_index + 1) * video_batch_size]
                        fb_labels = fb_labels_placeholder[gpu_index * image_batch_size:(gpu_index + 1) * image_batch_size, :]

                        loss_fT_videos, logits_fT_videos, loss_fb_images, logits_fb_images = create_architecture(scope, loss_fb_lst_dct, logits_fb_lst_dct, fb_dict, video_batch_size, image_batch_size, videos, images, fT_labels, fb_labels, dropout_placeholder)

                        logits_fT_videos_lst.append(logits_fT_videos)
                        loss_fT_videos_lst.append(loss_fT_videos)
                        logits_fb_images_lst.append(logits_fb_images)
                        loss_fb_images_lst.append(loss_fb_images)

                        varlist_fd, varlist_fT, varlist_fT_finetune, varlist_fT_main, varlist_fb = get_varlists()
                        grads_fd = opt_fd.compute_gradients(loss_fT_videos, varlist_fd)
                        grads_fT_main = opt_fT_main.compute_gradients(loss_fT_videos, varlist_fT_main)
                        grads_fT_finetune = opt_fT_finetune.compute_gradients(loss_fT_videos, varlist_fT_finetune)
                        grads_fb = opt_fb.compute_gradients(loss_fb_images, varlist_fb)

                        tower_grads_fd.append(grads_fd)
                        tower_grads_fb.append(grads_fb)
                        tower_grads_fT_main.append(grads_fT_main)
                        tower_grads_fT_finetune.append(grads_fT_finetune)

                        tf.get_variable_scope().reuse_variables()

        loss_fT_op = tf.reduce_mean(loss_fT_videos_lst, name='softmax')
        loss_fb_op = tf.reduce_mean(loss_fb_images_lst, name='softmax')

        logits_fT_op = tf.concat(logits_fT_videos_lst, 0)
        logits_fb_op = tf.concat(logits_fb_images_lst, 0)
        acc_fT_op = accuracy(logits_fT_op, fT_labels_placeholder)

        zero_ops_fd, accum_ops_fd, apply_gradient_op_fd = create_grad_accum_for_late_update(opt_fd, tower_grads_fd, varlist_fd, FLAGS.n_minibatches, global_step, decay_with_global_step=False)
        zero_ops_fb, accum_ops_fb, apply_gradient_op_fb = create_grad_accum_for_late_update(opt_fb, tower_grads_fb, varlist_fb, FLAGS.n_minibatches, global_step, decay_with_global_step=False)
        zero_ops_fT_finetune, accum_ops_fT_finetune, apply_gradient_op_fT_finetune = create_grad_accum_for_late_update(opt_fT_finetune, tower_grads_fT_finetune, varlist_fT_finetune, FLAGS.n_minibatches, global_step, decay_with_global_step=False)
        zero_ops_fT_main, accum_ops_fT_main, apply_gradient_op_fT_main = create_grad_accum_for_late_update(opt_fT_main, tower_grads_fT_main, varlist_fT_main, FLAGS.n_minibatches, global_step, decay_with_global_step=False)

        tr_videos_op, tr_videos_labels_op = create_videos_reading_ops(is_train=True, is_val=False,
                                                                      GPU_NUM=gpu_num, BATCH_SIZE=video_batch_size)
        val_videos_op, val_videos_labels_op = create_videos_reading_ops(is_train=False, is_val=True,
                                                                        GPU_NUM=gpu_num, BATCH_SIZE=video_batch_size)

        tr_images_op, tr_images_labels_op = create_images_reading_ops(is_train=True, is_val=False,
                                                                      GPU_NUM=gpu_num, BATCH_SIZE=image_batch_size)
        val_images_op, val_images_labels_op = create_images_reading_ops(is_train=False, is_val=True,
                                                                    GPU_NUM=gpu_num, BATCH_SIZE=image_batch_size)
        init_op = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())

        varlist_bn = [g for g in tf.global_variables() if 'moving_mean' in g.name]
        varlist_bn += [g for g in tf.global_variables() if 'moving_variance' in g.name]

        return (graph, init_op,
                zero_ops_fd, accum_ops_fd, apply_gradient_op_fd,
                zero_ops_fb, accum_ops_fb, apply_gradient_op_fb,
                zero_ops_fT_finetune, accum_ops_fT_finetune, apply_gradient_op_fT_finetune,
                zero_ops_fT_main, accum_ops_fT_main, apply_gradient_op_fT_main,
                loss_fb_op, logits_fb_op,
                loss_fT_op, logits_fT_op, acc_fT_op,
                tr_videos_op, tr_videos_labels_op,
                val_videos_op, val_videos_labels_op,
                tr_images_op, tr_images_labels_op,
                val_images_op, val_images_labels_op,
                videos_placeholder, images_placeholder, fT_labels_placeholder, fb_labels_placeholder, dropout_placeholder, isTraining_placeholder,
                varlist_fb, varlist_fT, varlist_fT_main, varlist_fT_finetune, varlist_fd, varlist_bn)

def update_fb(sess, step, n_minibatches, zero_fb_op, apply_gradient_fb_op, accum_fb_op, loss_fb_op,
              images_op, images_labels_op, images_placeholder, fb_images_labels_placeholder,
              dropout_placeholder, isTraining_placeholder):
    '''
    pretrain fb with fT=I
    '''
    start_time = time.time()
    sess.run(zero_fb_op)
    loss_fb_lst = []
    for _ in itertools.repeat(None, n_minibatches):
        images, images_labels = sess.run([images_op, images_labels_op])
        _, loss_fb = sess.run([accum_fb_op, loss_fb_op],
                                  feed_dict={images_placeholder: images,
                                             fb_images_labels_placeholder: images_labels,
                                             dropout_placeholder: 1.0,
                                             isTraining_placeholder: True})
        loss_fb_lst.append(loss_fb)
    sess.run([apply_gradient_fb_op])
    loss_summary = 'Step: {:4d}, time: {:.4f}, budget loss: {:.8f}'.format(
        step,
        time.time() - start_time, np.mean(loss_fb_lst))
    return loss_summary

def eval_fb(sess, step, n_minibatches, logits_fb_op, loss_fb_op, images_op, images_labels_op, images_placeholder, fb_images_labels_placeholder,
            dropout_placeholder, isTraining_placeholder):
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
                                                         dropout_placeholder: 1.0,
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

def update_fT_fd(sess, step, n_minibatches, zero_fd_op, apply_gradient_fd_op, accum_fd_op,
                 zero_fT_finetune_op, apply_gradient_fT_finetune_op, accum_fT_finetune_op,
                 zero_fT_main_op, apply_gradient_fT_main_op, accum_fT_main_op,
                 loss_fT_op, videos_op, videos_labels_op,
                 videos_placeholder, fT_videos_labels_placeholder, dropout_placeholder):
    start_time = time.time()
    sess.run([zero_fT_finetune_op, zero_fT_main_op, zero_fd_op])
    loss_fT_lst = []
    for _ in itertools.repeat(None, n_minibatches):
        videos, videos_labels = sess.run(
            [videos_op, videos_labels_op])
        _, _, _, loss_fT = sess.run([accum_fT_finetune_op, accum_fT_main_op, accum_fd_op, loss_fT_op],
                                    feed_dict={videos_placeholder: videos,
                                               fT_videos_labels_placeholder: videos_labels,
                                               dropout_placeholder: 1.0,
                                               })
        loss_fT_lst.append(loss_fT)
    sess.run([apply_gradient_fT_finetune_op, apply_gradient_fT_main_op, apply_gradient_fd_op])
    loss_summary = 'Updating fT + fd, Step: {:4d}, time: {:.4f}, fT loss: {:.8f}'.format(
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

def run_pretraining_fdfT():
    # Create model directory
    if not os.path.exists(FLAGS.pretrained_fdfT_ckpt_dir):
        os.makedirs(FLAGS.pretrained_fdfT_ckpt_dir)

    (graph, init_op,
     zero_fd_op, accum_fd_op, apply_gradient_fd_op,
     zero_fb_op, accum_fb_op, apply_gradient_fb_op,
     zero_fT_finetune_op, accum_fT_finetune_op, apply_gradient_fT_finetune_op,
     zero_fT_main_op, accum_fT_main_op, apply_gradient_fT_main_op,
     loss_fb_op, logits_fb_op,
     loss_fT_op, logits_fT_op, acc_fT_op,
     tr_videos_op, tr_videos_labels_op,
     val_videos_op, val_videos_labels_op,
     tr_images_op, tr_images_labels_op,
     val_images_op, val_images_labels_op,
     videos_placeholder, images_placeholder, fT_labels_placeholder, fb_labels_placeholder, dropout_placeholder,
     isTraining_placeholder,
     varlist_fb, varlist_fT, varlist_fT_main, varlist_fT_finetune, varlist_fd, varlist_bn) = build_graph(FLAGS.GPU_NUM, FLAGS.video_batch_size, FLAGS.image_batch_size)


    use_pretrained_model = True

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True

    with tf.Session(graph=graph, config=config) as sess:

        #saver = tf.train.Saver(tf.trainable_variables() + bn_moving_vars)
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # Create a saver for writing training checkpoints.
        if use_pretrained_model:
            varlist = [v for v in tf.trainable_variables() if
                       any(x in v.name.split('/')[0] for x in ["fd"])]
            restore_model_ckpt(sess, FLAGS.pretrained_fd_ckpt_dir, varlist, "fd")
            restore_model_pretrained_C3D(sess, COMMON_FLAGS.PRETRAINED_C3D_DIR, 'fT')
        else:
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir=FLAGS.pretrained_fdfT_ckpt_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Session restored from trained model at {}!'.format(ckpt.model_checkpoint_path))
            else:
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), FLAGS.pretrained_fdfT_ckpt_dir)

        saver = tf.train.Saver()
        for step in range(FLAGS.pretraining_steps_fdfT):
            loss_summary = update_fT_fd(sess, step, FLAGS.n_minibatches, zero_fd_op, apply_gradient_fd_op, accum_fd_op,
                         zero_fT_finetune_op, apply_gradient_fT_finetune_op, accum_fT_finetune_op,
                         zero_fT_main_op, apply_gradient_fT_main_op, accum_fT_main_op,
                         loss_fT_op, tr_videos_op, tr_videos_labels_op,
                         videos_placeholder, fT_labels_placeholder, dropout_placeholder)
            print("Updating fT + fd, "+loss_summary)

            if step % FLAGS.val_step == 0:

                eval_summary = eval_fT(sess, step, 30, loss_fT_op, acc_fT_op, tr_videos_op, tr_videos_labels_op,
                            videos_placeholder, fT_labels_placeholder, dropout_placeholder)
                print("TRAINING: "+eval_summary)

                eval_summary = eval_fT(sess, step, 30, loss_fT_op, acc_fT_op, val_videos_op, val_videos_labels_op,
                            videos_placeholder, fT_labels_placeholder, dropout_placeholder)
                print("VALIDATION: "+eval_summary)

            if step % FLAGS.save_step == 0 or (step + 1) == FLAGS.pretraining_steps_fdfT:
                checkpoint_path = os.path.join(FLAGS.pretrained_fdfT_ckpt_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

        coord.request_stop()
        coord.join(threads)

    print("done")

def run_pretraining_fb():
    '''
    
    '''
    # Create model directory
    if not os.path.exists(FLAGS.pretrained_fbfdfT_ckpt_dir):
        os.makedirs(FLAGS.pretrained_fbfdfT_ckpt_dir)

    (graph, init_op,
     zero_fd_op, accum_fd_op, apply_gradient_fd_op,
     zero_fb_op, accum_fb_op, apply_gradient_fb_op,
     zero_fT_finetune_op, accum_fT_finetune_op, apply_gradient_fT_finetune_op,
     zero_fT_main_op, accum_fT_main_op, apply_gradient_fT_main_op,
     loss_fb_op, logits_fb_op,
     loss_fT_op, logits_fT_op, acc_fT_op,
     tr_videos_op, tr_videos_labels_op,
     val_videos_op, val_videos_labels_op,
     tr_images_op, tr_images_labels_op,
     val_images_op, val_images_labels_op,
     videos_placeholder, images_placeholder, fT_labels_placeholder, fb_labels_placeholder, dropout_placeholder,
     isTraining_placeholder,
     varlist_fb, varlist_fT, varlist_fT_main, varlist_fT_finetune, varlist_fd, varlist_bn) = build_graph(FLAGS.GPU_NUM, FLAGS.video_batch_size, FLAGS.image_batch_size)


    use_pretrained_model = True

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True

    with tf.Session(graph=graph, config=config) as sess:

        #saver = tf.train.Saver(tf.trainable_variables() + bn_moving_vars)
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # Create a saver for writing training checkpoints.
        if use_pretrained_model:
            restore_model_ckpt(sess, FLAGS.pretrained_fdfT_ckpt_dir, varlist_fd+varlist_fT, "fd+fT")
        else:
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir=FLAGS.pretrained_fbfdfT_ckpt_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Session restored from trained model at {}!'.format(ckpt.model_checkpoint_path))
            else:
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), FLAGS.pretrained_fbfdfT_ckpt_dir)

        saver = tf.train.Saver()

        for step in range(FLAGS.pretraining_steps_fbfdfT):

            loss_summary = update_fb(sess, step, FLAGS.n_minibatches, zero_fb_op, apply_gradient_fb_op, accum_fb_op, loss_fb_op,
                      tr_images_op, tr_images_labels_op, images_placeholder, fb_labels_placeholder,
                      dropout_placeholder, isTraining_placeholder)
            print("Updating fb, " + loss_summary)

            if step % FLAGS.val_step == 0:
                eval_summary = eval_fb(sess, step, 30, logits_fb_op, loss_fb_op, tr_images_op, tr_images_labels_op,
                                       images_placeholder, fb_labels_placeholder, dropout_placeholder, isTraining_placeholder)
                print("TRAINING: " + eval_summary)
                eval_summary = eval_fb(sess, step, 30, logits_fb_op, loss_fb_op, val_images_op, val_images_labels_op,
                                       images_placeholder, fb_labels_placeholder, dropout_placeholder,
                                       isTraining_placeholder)
                print("VALIDATION: " + eval_summary)

            if step % FLAGS.save_step == 0 or (step + 1) == FLAGS.pretraining_steps_fbfdfT:
                checkpoint_path = os.path.join(FLAGS.pretrained_fbfdfT_ckpt_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    #run_pretraining_fdfT()
    run_pretraining_fb()
