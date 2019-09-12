import tensorflow as tf

def _instance_norm(X):
    batch_size, depth, height, width, nchannel = X.get_shape()
    X = tf.reshape(X, [batch_size * depth, height, width, nchannel])
    mean, variance = tf.nn.moments(X, axes=[1,2], keep_dims=True)
    epsilon = 1e-5
    inv = tf.rsqrt(variance + epsilon)
    normalized = (X-mean)*inv

    return tf.reshape(normalized, [batch_size, depth, height, width, nchannel])

def _bilinear_resize(X, factor=1):
    batch_size, depth, height, width, nchannel = X.get_shape()
    X = tf.reshape(X, [batch_size * depth, height, width, nchannel])
    X = tf.image.resize_bilinear(X, [height // factor, width // factor])
    X = tf.image.resize_bilinear(X, [height, width])
    return tf.reshape(X, [batch_size, depth, height, width, nchannel])

def _binary_activation(X):
    X = tf.reduce_mean(X, axis=4, keep_dims=True)
    X = tf.nn.relu(tf.sign(X))
    X = tf.tile(X, [1, 1, 1, 1, 3])
    return X

def _avg_replicate(X):
    X = tf.reduce_mean(X, axis=4, keep_dims=True)
    X = tf.tile(X, [1, 1, 1, 1, 3])
    return X