import errno, os 
import tensorflow as tf

import input_data
from loss import average_gradients

from common_flags import COMMON_FLAGS

def restore_model_ckpt(sess, ckpt_dir, varlist):
    '''
    Restore model only with give variable list
    Args:
        ckpt_dir: Either an dir containing .ckpt.data, .ckpt.meta, .ckpt.index or a .ckpt file
        varlist: which variables to restore
    '''
    if os.path.isfile(ckpt_dir): # when ckpt_dir is actually a .ckpt file
        saver = tf.train.Saver(varlist)
        saver.restore(sess, ckpt_dir)
        print('Session restored from pretrained model at {}!'.format(ckpt_dir))
    else: # when ckpt_dir is an dir containing .ckpt.data, .ckpt.meta, .ckpt.index
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir=ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver = tf.train.Saver(varlist)
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Session restored from pretrained model at {}!'.format(ckpt.model_checkpoint_path))
        else:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), ckpt_dir)

def accuracy(logits, labels):
    '''
    Compute the accuracy given logits and labels
    '''
    correct_pred = tf.equal(tf.argmax(tf.nn.softmax(logits), axis=1), labels)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return accuracy

def correct_num(logits, labels):
    '''
    Compute the correct_num given logits and labels
    '''
    correct_num = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(tf.nn.softmax(logits), axis=1), labels), tf.int32))
    return correct_num

def placeholder_inputs(batch_size):
    '''
    Create the placeholder ops in the graph
    Args:
        batch_size: #samples on one GPU * GPU_NUM
    Returns:
        5 placeholders: videos_placeholder, action_labels_placeholder, actor_labels_placeholder, dropout_placeholder, istraining_placeholder
    '''
    videos_placeholder = tf.placeholder(tf.float32, shape=(batch_size, COMMON_FLAGS.DEPTH, COMMON_FLAGS.CROP_HEIGHT, COMMON_FLAGS.CROP_WIDTH, COMMON_FLAGS.NCHANNEL))
    action_labels_placeholder = tf.placeholder(tf.int64, shape=(batch_size))
    actor_labels_placeholder = tf.placeholder(tf.int64, shape=(batch_size))
    dropout_placeholder = tf.placeholder(tf.float32)
    istraining_placeholder = tf.placeholder(tf.bool)
    return videos_placeholder, action_labels_placeholder, actor_labels_placeholder, dropout_placeholder, istraining_placeholder

def create_grad_accum_for_late_update(opt, tower_grads, tvarlist, n_minibatches, global_step, decay_with_global_step=False):
    '''
    Doing Late update: accumulate gradients and late update the parameters.
    Args:
        opt: optmization method. 
    '''
    # Average grads over GPUs (towers):
    grads = average_gradients(tower_grads)

    # accum_vars are variable storing the accumulated gradient
    # zero_ops are used to zero the accumulated gradient (comparable to optimizer.zero_grad() in PyTorch)
    with tf.device('/cpu:%d' % 0):
        accum_vars = [tf.Variable(tf.zeros_like(tvar.initialized_value()), trainable=False) for tvar in
                                    tvarlist]
        zero_ops = [tvar.assign(tf.zeros_like(tvar)) for tvar in accum_vars]

    accum_ops = [accum_vars[i].assign_add(gv[0] / n_minibatches) for i, gv in enumerate(grads)]

    if decay_with_global_step:
        global_increment = global_step.assign_add(1)
        with tf.control_dependencies([global_increment]):
            apply_gradient_op = opt.apply_gradients(
                [(accum_vars[i].value(), gv[1]) for i, gv in enumerate(grads)], global_step=None)
    else:
        apply_gradient_op = opt.apply_gradients(
            [(accum_vars[i].value(), gv[1]) for i, gv in enumerate(grads)], global_step=None)


    return zero_ops, accum_ops, apply_gradient_op

def create_videos_reading_ops(is_train, is_val, GPU_NUM, BATCH_SIZE):
    '''
    Multi-thread data fetching from queue
    Agrs:
        GPU_NUM: int. How many GPUs to paralize
        BATCH_SIZE: How many samples on a single GPU.
    '''
    train_files = [os.path.join(COMMON_FLAGS.TRAIN_FILES_DIR, f) for f in
                    os.listdir(COMMON_FLAGS.TRAIN_FILES_DIR) if f.endswith('.tfrecords')]
    val_files = [os.path.join(COMMON_FLAGS.VAL_FILES_DIR, f) for f in
                    os.listdir(COMMON_FLAGS.VAL_FILES_DIR) if f.endswith('.tfrecords')]
    test_files = [os.path.join(COMMON_FLAGS.TEST_FILES_DIR, f) for f in
                    os.listdir(COMMON_FLAGS.TEST_FILES_DIR) if f.endswith('.tfrecords')]

    num_threads = COMMON_FLAGS.NUM_THREADS
    num_examples_per_epoch = COMMON_FLAGS.NUM_EXAMPLES_PER_EPOCH
    if is_train:
        batch_size = BATCH_SIZE * GPU_NUM
        videos_op, action_labels_op, actor_labels_op = input_data.inputs_videos(filenames=train_files,
                                                                                batch_size=batch_size,
                                                                                num_epochs=None,
                                                                                num_threads=num_threads,
                                                                                num_examples_per_epoch=num_examples_per_epoch,
                                                                                shuffle=True)
    elif is_val:
        batch_size = BATCH_SIZE * GPU_NUM
        videos_op, action_labels_op, actor_labels_op = input_data.inputs_videos(filenames=val_files,
                                                                                batch_size=batch_size,
                                                                                num_epochs=1,
                                                                                num_threads=num_threads,
                                                                                num_examples_per_epoch=num_examples_per_epoch,
                                                                                shuffle=False)
    else:
        batch_size = BATCH_SIZE * GPU_NUM
        videos_op, action_labels_op, actor_labels_op = input_data.inputs_videos(filenames=test_files,
                                                                                batch_size=batch_size,
                                                                                num_epochs=1,
                                                                                num_threads=num_threads,
                                                                                num_examples_per_epoch=num_examples_per_epoch,
                                                                                shuffle=False)

    return videos_op, action_labels_op, actor_labels_op

def create_images_reading_ops(is_train, is_val, GPU_NUM, BATCH_SIZE):
    '''
    Only used by pretraining fd. Still validating on one batch, not the whole dataset.
    Agrs:
        GPU_NUM: int. How many GPUs to paralize
        BATCH_SIZE: How many samples on a single GPU.
    '''
    train_files = [os.path.join(COMMON_FLAGS.TRAIN_FILES_DEG_DIR, f) for f in
                    os.listdir(COMMON_FLAGS.TRAIN_FILES_DEG_DIR) if f.endswith('.tfrecords')]
    val_files = [os.path.join(COMMON_FLAGS.VAL_FILES_DEG_DIR, f) for f in
                    os.listdir(COMMON_FLAGS.VAL_FILES_DEG_DIR) if f.endswith('.tfrecords')]

    num_threads = COMMON_FLAGS.NUM_THREADS
    num_examples_per_epoch = COMMON_FLAGS.NUM_EXAMPLES_PER_EPOCH
    batch_size = BATCH_SIZE * GPU_NUM

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

def avg_replicate(X):
    '''
    Find average across the channel dimenssion i.e. 4th dimension of (N,T,H,W,C) tensor, and then replicate along the channel dimenssion.
    '''
    X = tf.reduce_mean(X, axis=4, keepdims=True)
    X = tf.tile(X, [1, 1, 1, 1, 3])
    return X