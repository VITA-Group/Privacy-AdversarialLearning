import numpy as np
import tensorflow as tf

# MSE loss
def tower_loss_mse(name_scope, preds, labels):
    labels = tf.cast(labels, tf.float32)
    MSE = tf.reduce_mean(
            tf.square(labels - preds))
    tf.summary.scalar(
            name_scope + 'mse',
            MSE)
    return MSE


# Cross entropy with dense (1-hot) label
def tower_loss_xentropy_dense(logits, labels):
    labels = tf.cast(labels, tf.float32)
    xentropy_mean = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    )
    return xentropy_mean

# Cross entropy with sparse (scalar) label
def tower_loss_xentropy_sparse(name_scope, logits, labels, use_weight_decay=False):
    labels = tf.cast(labels, tf.int64)
    xentropy_mean = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))

    if use_weight_decay:
        tf.add_to_collection('c3d_losses', xentropy_mean)
        losses = tf.get_collection('c3d_losses', name_scope)
        return tf.add_n(losses, name='c3d_losses')
    return xentropy_mean

# Cross entropy with uniform label
def tower_loss_xentropy_uniform(logits):
    batch_size = logits.get_shape()[0]
    num_classes = 13
    uniform_labels = np.full((batch_size, num_classes), 1 / num_classes, dtype=np.float32)
    uniform_labels = tf.convert_to_tensor(uniform_labels, np.float32)
    xentropy_mean = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=uniform_labels))
    return xentropy_mean

# Max cross entropy with uniform label
def tower_loss_max_xentropy_uniform(logits_lst):
    batch_size = logits_lst[0].get_shape()[0]
    num_classes = 13
    xentropy_uniform_tensor_lst = []
    uniform_labels = np.full((batch_size, num_classes),
                                   1 / num_classes, dtype=np.float32)
    uniform_labels = tf.convert_to_tensor(uniform_labels, np.float32)
    for logits in logits_lst:
        xentropy_uniform_tensor = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=uniform_labels)
        xentropy_uniform_tensor_lst.append(xentropy_uniform_tensor)
    xentropy_uniform_tensor_stack = tf.stack(xentropy_uniform_tensor_lst, axis=0)
    argmax_xentropy_uniform = tf.argmax(xentropy_uniform_tensor_stack, axis=0)
    max_xentropy_uniform = tf.reduce_mean(tf.reduce_max(xentropy_uniform_tensor_stack, axis=0))
    return max_xentropy_uniform, argmax_xentropy_uniform

# Negative entropy
def tower_loss_neg_entropy(logits):
    softmax = tf.nn.softmax(logits)
    nentropy = tf.reduce_sum(tf.multiply(softmax, tf.log(softmax)))
    return nentropy

# Max negative entropy
def tower_loss_max_neg_entropy(logits_lst):
    nentropy_tensor_lst = []
    for logits in logits_lst:
        softmax = tf.nn.softmax(logits)
        nentropy_tensor = tf.reduce_sum(tf.multiply(softmax, tf.log(softmax)), axis=1)
        nentropy_tensor_lst.append(nentropy_tensor)
    nentropy_tensor_stack = tf.stack(nentropy_tensor_lst, axis=0)
    argmax_nentropy = tf.argmax(nentropy_tensor_stack, axis=0)
    max_nentropy = tf.reduce_mean(tf.reduce_max(nentropy_tensor_stack, axis=0))
    return max_nentropy, argmax_nentropy


# Average the gradients for each shared variable across all towers
def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
        List of pairs of (gradient, variable) where the gradient has been averaged
        across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads