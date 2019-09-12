import tensorflow as tf
import sys
from os.path import dirname, realpath
sys.path.append(dirname(realpath(__file__)))
from common_flags import COMMON_FLAGS
import re
import tensorflow.contrib.slim as slim
from collections import namedtuple

def fb(images, is_training=True, depth_multiplier=1.0, min_depth=8, test=False):
    depth = lambda d: max(int(d * depth_multiplier), min_depth)
    batch_norm_params = {
        'center': True,
        'scale': True,
        'decay': 0.999,
        'epsilon': 0.001,
        'is_training': is_training,
    }

    # Set weight_decay for weights in Conv and DepthSepConv layers.
    weights_init = tf.truncated_normal_initializer(stddev=0.09)
    regularizer = tf.contrib.layers.l2_regularizer(0.00004)
    depthwise_regularizer = regularizer

    Conv = namedtuple('Conv', ['kernel', 'stride', 'depth'])
    DepthSepConv = namedtuple('DepthSepConv', ['kernel', 'stride', 'depth'])

    # _CONV_DEFS specifies the MobileNet body
    _CONV_DEFS = [
        Conv(kernel=[3, 3], stride=2, depth=32),
        DepthSepConv(kernel=[3, 3], stride=1, depth=64),
        DepthSepConv(kernel=[3, 3], stride=2, depth=128),
        DepthSepConv(kernel=[3, 3], stride=1, depth=128),
        DepthSepConv(kernel=[3, 3], stride=2, depth=256),
        DepthSepConv(kernel=[3, 3], stride=1, depth=256),
        DepthSepConv(kernel=[3, 3], stride=2, depth=512),
        DepthSepConv(kernel=[3, 3], stride=1, depth=512),
        DepthSepConv(kernel=[3, 3], stride=1, depth=512),
        DepthSepConv(kernel=[3, 3], stride=1, depth=512),
        DepthSepConv(kernel=[3, 3], stride=1, depth=512),
        DepthSepConv(kernel=[3, 3], stride=1, depth=512),
        DepthSepConv(kernel=[3, 3], stride=2, depth=1024),
        DepthSepConv(kernel=[3, 3], stride=1, depth=1024)
    ]

    with tf.variable_scope('fb_{}'.format(depth_multiplier)):
        with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                            weights_initializer=weights_init, activation_fn=tf.nn.relu6, normalizer_fn=slim.batch_norm):
            with slim.arg_scope([slim.batch_norm], **batch_norm_params):
                with slim.arg_scope([slim.conv2d, slim.separable_conv2d], padding='SAME'):
                    with slim.arg_scope([slim.conv2d], weights_regularizer=regularizer):
                        with slim.arg_scope([slim.separable_conv2d], weights_regularizer=depthwise_regularizer):
                            net = images
                            for i, conv_def in enumerate(_CONV_DEFS):
                                end_point_base = 'Conv2d_%d' % i
                                if isinstance(conv_def, Conv):
                                    end_point = end_point_base
                                    net = slim.conv2d(net, depth(conv_def.depth), conv_def.kernel,
                                                  stride=conv_def.stride,
                                                  normalizer_fn=slim.batch_norm,
                                                  scope=end_point)
                                elif isinstance(conv_def, DepthSepConv):
                                    end_point = end_point_base + '_depthwise'
                                    net = slim.separable_conv2d(net, None, conv_def.kernel,
                                                            depth_multiplier=1,
                                                            stride=conv_def.stride,
                                                            normalizer_fn=slim.batch_norm,
                                                            scope=end_point)
                                    end_point = end_point_base + '_pointwise'
                                    net = slim.conv2d(net, depth(conv_def.depth), [1, 1],
                                                  stride=1,
                                                  normalizer_fn=slim.batch_norm,
                                                  scope=end_point)
                                else:
                                    raise ValueError('Unknown convolution type %s for layer %d'
                                                 % (conv_def.ltype, i))
        with tf.variable_scope('Logits'):
            net = tf.reduce_mean(net, [1, 2], keepdims=True, name='global_pool')
            logits = slim.conv2d(net, COMMON_FLAGS.NUM_CLASSES_BUDGET, [1, 1], activation_fn=None, normalizer_fn=None, scope='Conv2d_1c_1x1')
            logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')
    return logits