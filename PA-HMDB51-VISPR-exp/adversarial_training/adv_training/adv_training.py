import sys, time, os, datetime, errno, pprint, itertools
from os.path import dirname, realpath
sys.path.append(dirname(dirname(dirname(realpath(__file__)))))

import tensorflow as tf
from common_flags import COMMON_FLAGS
from adversarial_training.adv_training.adv_flags import FLAGS
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
    with tf.variable_scope(tf.get_variable_scope(), reuse=False):
        fd_videos = fd(videos)
    fT_videos = tf.reshape(fd_videos, [video_batch_size, COMMON_FLAGS.DEPTH, COMMON_FLAGS.CROP_HEIGHT, COMMON_FLAGS.CROP_WIDTH, COMMON_FLAGS.NCHANNEL])
    logits_fT = fT(fT_videos, dropout)
    loss_fT = tower_loss_xentropy_sparse(
        logits_fT,
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
    loss_fb_images = tf.divide(loss_fb_images, 4.0, 'LossFbMean')
    logits_fb_images = tf.divide(logits_fb_images, 4.0, 'LogitsFbMean')

    loss_fb_videos = 0.0
    loss_fb_videos_lst = []
    fb_uniform_labels = np.full((video_batch_size, COMMON_FLAGS.NUM_CLASSES_BUDGET),
                                     1, dtype=np.float32)
    fb_uniform_labels = tf.convert_to_tensor(fb_uniform_labels, np.float32)

    for name, fb in fb_dict.items():
        print(name)
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            logits, _ = fb(fd_videos)
        logits_fb_videos = tf.reshape(logits, [-1, COMMON_FLAGS.DEPTH, COMMON_FLAGS.NUM_CLASSES_BUDGET])
        logits_fb_videos = tf.reduce_mean(logits_fb_videos, axis=1, keepdims=False)
        loss_tensor = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=logits_fb_videos, labels=fb_uniform_labels)
        loss = tf.reduce_mean(loss_tensor)
        loss_tensor = tf.reshape(tf.reduce_mean(loss_tensor, axis=1), [-1])
        loss_fb_videos += loss
        loss_fb_videos_lst.append(loss_tensor)

    loss_fb_videos_tensor_stack = tf.stack(loss_fb_videos_lst, axis=0)
    argmax_centpy = tf.argmax(loss_fb_videos_tensor_stack, axis=0)
    max_centpy = tf.reduce_mean(tf.reduce_max(loss_fb_videos_tensor_stack, axis=0))

    loss_fd = loss_fT + FLAGS.gamma * max_centpy

    return loss_fT, logits_fT, loss_fb_images, logits_fb_images, loss_fd, argmax_centpy

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
                            FLAGS.video_batch_size * FLAGS.GPU_NUM, FLAGS.image_batch_size * FLAGS.GPU_NUM)


        tower_grads_fd, tower_grads_fT, tower_grads_fb = [], [], []

        logits_fT_videos_lst, logits_fb_images_lst = [], []

        loss_fT_videos_lst, loss_fb_images_lst, loss_fd_lst = [], [], []
        argmax_centpy_lst = []
        opt_fd = tf.train.AdamOptimizer(1e-4)
        opt_fT = tf.train.AdamOptimizer(1e-5)
        opt_fb = tf.train.AdamOptimizer(1e-4)

        fb_name_dict = {4: ['resnet_v1_50', 'resnet_v2_50', 'mobilenet_v1', 'mobilenet_v1_075'],
                        3: ['resnet_v1_50', 'resnet_v2_50', 'mobilenet_v1'],
                        2: ['resnet_v1_50', 'resnet_v2_50'],
                        1: ['resnet_v1_50']}
        fb_name_lst = fb_name_dict[FLAGS.M]
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
                        utility_labels = fT_labels_placeholder[gpu_index * video_batch_size:(gpu_index + 1) * video_batch_size]
                        budget_labels = fb_labels_placeholder[gpu_index * image_batch_size:(gpu_index + 1) * image_batch_size, :]

                        loss_fT_videos, logits_fT_videos, loss_fb_images, logits_fb_images, loss_fd, argmax_centpy = create_architecture(scope, loss_fb_lst_dct, logits_fb_lst_dct, fb_dict, video_batch_size, image_batch_size, videos, images, utility_labels, budget_labels, dropout_placeholder)

                        logits_fT_videos_lst.append(logits_fT_videos)
                        loss_fT_videos_lst.append(loss_fT_videos)
                        logits_fb_images_lst.append(logits_fb_images)
                        loss_fb_images_lst.append(loss_fb_images)
                        loss_fd_lst.append(loss_fd)
                        argmax_centpy_lst.append(argmax_centpy)

                        varlist_fd, varlist_fT, _, _, varlist_fb = get_varlists()
                        grads_fd = opt_fd.compute_gradients(loss_fd, varlist_fd)
                        grads_fT = opt_fT.compute_gradients(loss_fT_videos, varlist_fT+varlist_fd)
                        grads_fb = opt_fb.compute_gradients(loss_fb_images, varlist_fb)

                        tower_grads_fd.append(grads_fd)
                        tower_grads_fb.append(grads_fb)
                        tower_grads_fT.append(grads_fT)

                        tf.get_variable_scope().reuse_variables()

        argmax_centpy_op = tf.concat(argmax_centpy_lst, 0)
        loss_fT_op = tf.reduce_mean(loss_fT_videos_lst, name='softmax')
        loss_fb_op = tf.reduce_mean(loss_fb_images_lst, name='softmax')
        loss_fd_op = tf.reduce_mean(loss_fd_lst, name='softmax')


        logits_fT_op = tf.concat(logits_fT_videos_lst, 0)
        logits_fb_op = tf.concat(logits_fb_images_lst, 0)
        acc_fT_op = accuracy(logits_fT_op, fT_labels_placeholder)

        zero_fd_op, accum_fd_op, apply_gradient_fd_op = create_grad_accum_for_late_update(opt_fd, tower_grads_fd, varlist_fd, FLAGS.n_minibatches, global_step, decay_with_global_step=False)
        zero_fb_op, accum_fb_op, apply_gradient_fb_op = create_grad_accum_for_late_update(opt_fb, tower_grads_fb, varlist_fb, FLAGS.n_minibatches, global_step, decay_with_global_step=False)
        zero_fT_op, accum_fT_op, apply_gradient_fT_op = create_grad_accum_for_late_update(opt_fT, tower_grads_fT, varlist_fT+varlist_fd, FLAGS.n_minibatches, global_step, decay_with_global_step=False)

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
                zero_fd_op, accum_fd_op, apply_gradient_fd_op,
                zero_fb_op, accum_fb_op, apply_gradient_fb_op,
                zero_fT_op, accum_fT_op, apply_gradient_fT_op,
                loss_fb_op, logits_fb_op,
                loss_fT_op, logits_fT_op, acc_fT_op,
                loss_fd_op, argmax_centpy_op,
                tr_videos_op, tr_videos_labels_op,
                val_videos_op, val_videos_labels_op,
                tr_images_op, tr_images_labels_op,
                val_images_op, val_images_labels_op,
                videos_placeholder, images_placeholder, fT_labels_placeholder, fb_labels_placeholder, dropout_placeholder, isTraining_placeholder,
                varlist_fb, varlist_fT, varlist_fd, varlist_bn)

def update_fb(sess, step, n_minibatches, zero_fb_op, apply_gradient_fb_op, accum_fb_op, loss_fb_op,
              images_op, images_labels_op, images_placeholder, fb_images_labels_placeholder, isTraining_placeholder):
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
            images_placeholder, fb_images_labels_placeholder,isTraining_placeholder):
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

def update_fd(sess, step, n_minibatches, zero_fd_op, apply_gradient_fd_op, accum_fd_op, loss_fd_op,
              loss_fT_op, argmax_centpy_op, videos_op, videos_labels_op,
              videos_placeholder, fT_videos_labels_placeholder, dropout_placeholder, isTraining_placeholder):
    start_time = time.time()
    loss_fd_lst, loss_fT_lst = [], []
    sess.run(zero_fd_op)
    for _ in itertools.repeat(None, n_minibatches):
        videos, videos_labels = sess.run([videos_op, videos_labels_op])
        _, argmax_cent, loss_fd, loss_fT = sess.run(
            [accum_fd_op, argmax_centpy_op, loss_fd_op, loss_fT_op],
            feed_dict={videos_placeholder: videos,
                       fT_videos_labels_placeholder: videos_labels,
                       dropout_placeholder: 1.0,
                       isTraining_placeholder: True})
        print(argmax_cent)
        loss_fd_lst.append(loss_fd)
        loss_fT_lst.append(loss_fT)
    sess.run(apply_gradient_fd_op)
    loss_summary = 'Step: {:4d}, time: {:.4f}, fd loss: {:.8f}, fT loss: {:.8f}'.format(
        step,
        time.time() - start_time, np.mean(loss_fd_lst), np.mean(loss_fT_lst))
    return loss_summary

def update_fT(sess, step, n_minibatches, zero_fT_op, apply_gradient_fT_op, accum_fT_op, loss_fd_op,
              loss_fT_op, videos_op, videos_labels_op, videos_placeholder, fT_videos_labels_placeholder,
              dropout_placeholder, isTraining_placeholder):
    start_time = time.time()
    sess.run(zero_fT_op)
    loss_fd_lst, loss_fT_lst = [], []
    for _ in itertools.repeat(None, n_minibatches):
        videos, videos_labels = sess.run([videos_op, videos_labels_op])
        _, loss_fd, loss_fT = sess.run([accum_fT_op, loss_fd_op, loss_fT_op],
            feed_dict={videos_placeholder: videos,
                       fT_videos_labels_placeholder: videos_labels,
                       dropout_placeholder: 1.0,
                       isTraining_placeholder: True})
        loss_fd_lst.append(loss_fd)
        loss_fT_lst.append(loss_fT)
    sess.run([apply_gradient_fT_op])
    loss_summary = 'Step: {:4d}, time: {:.4f}, fd loss: {:.8f}, fT loss: {:.8f}'.format(
        step,
        time.time() - start_time, np.mean(loss_fd_lst), np.mean(loss_fT_lst))
    return loss_summary

def eval_fT(sess, step, n_minibatches, loss_fd_op, loss_fT_op, acc_fT_op, videos_op, videos_labels_op,
            videos_placeholder, fT_videos_labels_placeholder, dropout_placeholder, isTraining_placeholder):
    start_time = time.time()
    acc_fT_lst, loss_fd_lst, loss_fT_lst = [], [], []
    for _ in itertools.repeat(None, n_minibatches):
        videos, videos_labels = sess.run(
            [videos_op, videos_labels_op])
        acc_fT, loss_fd, loss_fT = sess.run(
            [acc_fT_op, loss_fd_op, loss_fT_op],
            feed_dict={videos_placeholder: videos,
                       fT_videos_labels_placeholder: videos_labels,
                       dropout_placeholder: 1.0,
                       isTraining_placeholder: True,
                       })
        acc_fT_lst.append(acc_fT)
        loss_fd_lst.append(loss_fd)
        loss_fT_lst.append(loss_fT)
    eval_summary = "Step: {:4d}, time: {:.4f}, fd loss: {:.8f}, fT loss: {:.8f}, fT accuracy: {:.5f},\n".format(
                    step, time.time() - start_time, np.mean(loss_fd_lst),
                    np.mean(loss_fT_lst), np.mean(acc_fT_lst))
    return eval_summary, np.mean(acc_fT_lst)

def run_adversarial_training():
    # Create model directory
    if not os.path.exists(FLAGS.adv_ckpt_dir):
        os.makedirs(FLAGS.adv_ckpt_dir)

    (graph, init_op,
     zero_fd_op, accum_fd_op, apply_gradient_fd_op,
     zero_fb_op, accum_fb_op, apply_gradient_fb_op,
     zero_fT_op, accum_fT_op, apply_gradient_fT_op,
     loss_fb_op, logits_fb_op,
     loss_fT_op, logits_fT_op, acc_fT_op,
     loss_fd_op, argmax_centpy_op,
     tr_videos_op, tr_videos_labels_op,
     val_videos_op, val_videos_labels_op,
     tr_images_op, tr_images_labels_op,
     val_images_op, val_images_labels_op,
     videos_placeholder, images_placeholder, fT_videos_labels_placeholder, fb_images_labels_placeholder, dropout_placeholder, isTraining_placeholder,
     varlist_fb, varlist_fT, varlist_fd, varlist_bn) = build_graph(FLAGS.GPU_NUM, FLAGS.video_batch_size, FLAGS.image_batch_size)


    use_pretrained_model = True

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True

    with tf.Session(graph=graph, config=config) as sess:
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        if use_pretrained_model:
            saver = tf.train.Saver(varlist_fT + varlist_fd + varlist_fb)
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir=FLAGS.pretrained_fbfdfT_ckpt_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Session restored from trained model at {}!'.format(ckpt.model_checkpoint_path))
            else:
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), FLAGS.pretrained_fbfdfT_ckpt_dir)
        else:
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir=FLAGS.adv_ckpt_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Session restored from trained model at {}!'.format(ckpt.model_checkpoint_path))
            else:
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), FLAGS.adv_ckpt_dir)

        gvar_list = tf.global_variables()
        bn_moving_vars = [g for g in gvar_list if 'moving_mean' in g.name]
        bn_moving_vars += [g for g in gvar_list if 'moving_variance' in g.name]
        # saver = tf.train.Saver(tf.trainable_variables() + bn_moving_vars)
        if not os.path.exists(FLAGS.summary_dir):
            os.makedirs(FLAGS.summary_dir)
        loss_summary_file = open(FLAGS.summary_dir + 'loss_summary.txt', 'w')
        train_summary_file = open(FLAGS.summary_dir + 'train_summary.txt', 'w')
        test_summary_file = open(FLAGS.summary_dir + 'test_summary.txt', 'w')

        ckpt_saver = tf.train.Saver()
        for step in range(FLAGS.max_steps):
            if step == 0 or (FLAGS.use_fb_restarting and step % FLAGS.restarting_step == 0):
                saver = tf.train.Saver(varlist_fb)
                ckpt = tf.train.get_checkpoint_state(checkpoint_dir=FLAGS.pretrained_fbfdfT_ckpt_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    print('Session restored from trained model at {}!'.format(ckpt.model_checkpoint_path))
                else:
                    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), FLAGS.pretrained_fbfdfT_ckpt_dir)
                for step in range(FLAGS.pretraining_steps_fb):
                    loss_summary = update_fb(sess, step, FLAGS.n_minibatches, zero_fb_op, apply_gradient_fb_op, accum_fb_op, loss_fb_op,
                                             tr_images_op, tr_images_labels_op, images_placeholder, fb_images_labels_placeholder,
                                             isTraining_placeholder)
                    print("Pretraining / Restarting (fb), "+loss_summary)
                    if step % FLAGS.val_step == 0:
                        eval_summary = eval_fb(sess, step, 60, logits_fb_op, loss_fb_op, val_images_op, val_images_labels_op,
                                               images_placeholder, fb_images_labels_placeholder, isTraining_placeholder)
                        print("VALIDATION: Pretraining / Restarting (fb), "+eval_summary)
            loss_summary = update_fd(sess, step, FLAGS.n_minibatches, zero_fd_op, apply_gradient_fd_op, accum_fd_op, loss_fd_op,
                                     loss_fT_op, argmax_centpy_op, tr_videos_op, tr_videos_labels_op,
                                     videos_placeholder, fT_videos_labels_placeholder, dropout_placeholder,isTraining_placeholder)
            print("Alternating Training (fd), "+loss_summary)
            loss_summary_file.write(loss_summary + '\n')

            while True:
                eval_summary, acc_fT = eval_fT(sess, step, FLAGS.n_minibatches, loss_fd_op, loss_fT_op, acc_fT_op,
                                               val_videos_op, val_videos_labels_op, videos_placeholder, fT_videos_labels_placeholder,
                                               dropout_placeholder, isTraining_placeholder)
                print("Monitoring (fT), "+eval_summary)

                if acc_fT >= FLAGS.fT_acc_val_thresh:
                    break

                loss_summary = update_fT(sess, step, FLAGS.n_minibatches, zero_fT_op, apply_gradient_fT_op, accum_fT_op, loss_fd_op, loss_fT_op,
                                         tr_videos_op, tr_videos_labels_op, videos_placeholder, fT_videos_labels_placeholder,
                                         dropout_placeholder, isTraining_placeholder)

                print("Alternating Training (fT), "+loss_summary)


            for _ in itertools.repeat(None, 5):
                loss_summary = update_fb(sess, step, FLAGS.n_minibatches, zero_fb_op, apply_gradient_fb_op, accum_fb_op, loss_fb_op,
                                         tr_images_op, tr_images_labels_op, images_placeholder, fb_images_labels_placeholder,
                                         isTraining_placeholder)
                print("Alternating Training (fb), "+loss_summary)

            if step % FLAGS.val_step == 0:

                eval_summary, _ = eval_fT(sess, step, 60, loss_fd_op, loss_fT_op, acc_fT_op, tr_videos_op, tr_videos_labels_op,
                                          videos_placeholder, fT_videos_labels_placeholder, dropout_placeholder, isTraining_placeholder)
                print("TRAINING: "+eval_summary)
                train_summary_file.write(eval_summary + '\n')

                eval_summary, _ = eval_fT(sess, step, 60, loss_fd_op, loss_fT_op, acc_fT_op, val_videos_op, val_videos_labels_op,
                                          videos_placeholder, fT_videos_labels_placeholder, dropout_placeholder, isTraining_placeholder)
                print("VALIDATION: "+eval_summary)
                test_summary_file.write(eval_summary + '\n')

                eval_summary = eval_fb(sess, step, 60, logits_fb_op, loss_fb_op, val_images_op, val_images_labels_op,
                                       images_placeholder, fb_images_labels_placeholder, isTraining_placeholder)
                print("VALIDATION: "+eval_summary)
                test_summary_file.write(eval_summary + '\n')

            if step % FLAGS.save_step == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_path = os.path.join(FLAGS.adv_ckpt_dir, 'model.ckpt')
                ckpt_saver.save(sess, checkpoint_path, global_step=step)

        loss_summary_file.close()
        train_summary_file.close()
        test_summary_file.close()
        coord.request_stop()
        coord.join(threads)

    print("done")

if __name__ == '__main__':
    run_adversarial_training()