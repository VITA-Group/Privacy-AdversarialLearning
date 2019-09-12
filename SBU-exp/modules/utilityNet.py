import tensorflow as tf

def _variable_on_cpu(name, shape, initializer):
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer)
    return var

# The definition of utilityNet: f_T
# C3D Generic Feature for Video Analysis: http://vlg.cs.dartmouth.edu/c3d/
def utilityNet(X, wd=0.0005):
    def _variable_with_weight_decay(name, shape, wd):
        var = _variable_on_cpu(name, shape, tf.contrib.layers.xavier_initializer())
        if wd is not None:
            weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
            tf.add_to_collection('c3d_losses', weight_decay)
        return var
    def conv3d(name, l_input, w, b):
        return tf.nn.bias_add(
            tf.nn.conv3d(l_input, w, strides=[1, 1, 1, 1, 1], padding='SAME', name=name), b)

    def max_pool(name, l_input, k):
        return tf.nn.max_pool3d(l_input, ksize=[1, k, 2, 2, 1], strides=[1, k, 2, 2, 1], padding='SAME', name=name)

    with tf.variable_scope('UtilityModule'):
        weights = {
        'wc1': _variable_with_weight_decay('wc1', [3, 3, 3, 3, 64], wd),
        'wc2': _variable_with_weight_decay('wc2', [3, 3, 3, 64, 128], wd),
        'wc3a': _variable_with_weight_decay('wc3a', [3, 3, 3, 128, 256], wd),
        'wc3b': _variable_with_weight_decay('wc3b', [3, 3, 3, 256, 256], wd),
        'wc4a': _variable_with_weight_decay('wc4a', [3, 3, 3, 256, 512], wd),
        'wc4b': _variable_with_weight_decay('wc4b', [3, 3, 3, 512, 512], wd),
        'wc5a': _variable_with_weight_decay('wc5a', [3, 3, 3, 512, 512], wd),
        'wc5b': _variable_with_weight_decay('wc5b', [3, 3, 3, 512, 512], wd),
        'wd1': _variable_with_weight_decay('wd1', [8192, 4096], wd),
        'wd2': _variable_with_weight_decay('wd2', [4096, 4096], wd),
        'out': _variable_with_weight_decay('wout', [4096, 8], wd)
        }
        biases = {
        'bc1': _variable_with_weight_decay('bc1', [64], 0.000),
        'bc2': _variable_with_weight_decay('bc2', [128], 0.000),
        'bc3a': _variable_with_weight_decay('bc3a', [256], 0.000),
        'bc3b': _variable_with_weight_decay('bc3b', [256], 0.000),
        'bc4a': _variable_with_weight_decay('bc4a', [512], 0.000),
        'bc4b': _variable_with_weight_decay('bc4b', [512], 0.000),
        'bc5a': _variable_with_weight_decay('bc5a', [512], 0.000),
        'bc5b': _variable_with_weight_decay('bc5b', [512], 0.000),
        'bd1': _variable_with_weight_decay('bd1', [4096], 0.000),
        'bd2': _variable_with_weight_decay('bd2', [4096], 0.000),
        'out': _variable_with_weight_decay('bout', [8], 0.000),
        }
    # Convolution Layer
    conv1 = conv3d('conv1', X, weights['wc1'], biases['bc1'])
    conv1 = tf.nn.relu(conv1, 'relu1')
    pool1 = max_pool('pool1', conv1, k=1)

    # Convolution Layer
    conv2 = conv3d('conv2', pool1, weights['wc2'], biases['bc2'])
    conv2 = tf.nn.relu(conv2, 'relu2')
    pool2 = max_pool('pool2', conv2, k=2)

    # Convolution Layer
    conv3 = conv3d('conv3a', pool2, weights['wc3a'], biases['bc3a'])
    conv3 = tf.nn.relu(conv3, 'relu3a')
    conv3 = conv3d('conv3b', conv3, weights['wc3b'], biases['bc3b'])
    conv3 = tf.nn.relu(conv3, 'relu3b')
    pool3 = max_pool('pool3', conv3, k=2)

    # Convolution Layer
    conv4 = conv3d('conv4a', pool3, weights['wc4a'], biases['bc4a'])
    conv4 = tf.nn.relu(conv4, 'relu4a')
    conv4 = conv3d('conv4b', conv4, weights['wc4b'], biases['bc4b'])
    conv4 = tf.nn.relu(conv4, 'relu4b')
    pool4 = max_pool('pool4', conv4, k=2)

    # Convolution Layer
    conv5 = conv3d('conv5a', pool4, weights['wc5a'], biases['bc5a'])
    conv5 = tf.nn.relu(conv5, 'relu5a')
    conv5 = conv3d('conv5b', conv5, weights['wc5b'], biases['bc5b'])
    conv5 = tf.nn.relu(conv5, 'relu5b')
    pool5 = max_pool('pool5', conv5, k=2)

    # Fully connected layer
    dense1 = tf.reshape(pool5, [-1, weights['wd1'].get_shape().as_list()[0]])
    dense1 = tf.matmul(dense1, weights['wd1']) + biases['bd1']
    dense1 = tf.nn.relu(dense1, name='fc1')
    # dense1 = tf.nn.dropout(dense1, _dropout)

    dense2 = tf.matmul(dense1, weights['wd2']) + biases['bd2']
    dense2 = tf.nn.relu(dense2, name='fc2') # Relu activation
    # dense2 = tf.nn.dropout(dense2, _dropout)

    # Output: class prediction
    softmax_linear = tf.matmul(dense2, weights['out']) + biases['out']
    return softmax_linear