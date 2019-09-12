from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import hopenet_utils
from tf_flags import FLAGS

resnet_arg_scope = hopenet_utils.resnet_arg_scope
slim = tf.contrib.slim

@slim.add_arg_scope
def bottleneck(inputs,
               depth,
               depth_bottleneck,
               stride,
               rate=1,
               outputs_collections=None,
               scope=None):
    with tf.variable_scope(scope, 'bottleneck_v1', [inputs]) as sc:
        depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
        if depth == depth_in:
            shortcut = hopenet_utils.subsample(inputs, stride, 'shortcut')
        else:
            shortcut = slim.conv2d(
                inputs,
                depth, [1, 1],
                stride=stride,
                activation_fn=None,
                scope='shortcut')

        residual = slim.conv2d(inputs, depth_bottleneck, [1, 1], stride=1, scope='conv1')
        residual = hopenet_utils.conv2d_same(residual, depth_bottleneck, 3, stride, rate=rate, scope='conv2')
        residual = slim.conv2d(residual, depth, [1, 1], stride=1, activation_fn=None, scope='conv3')


        output = tf.nn.relu(shortcut + residual)

        return slim.utils.collect_named_outputs(outputs_collections, sc.name, output)


def resnet_v1_block(scope, base_depth, num_units, stride):
  return hopenet_utils.Block(scope, bottleneck, [{
      'depth': base_depth * 4,
      'depth_bottleneck': base_depth,
      'stride': stride
  }] + [{
      'depth': base_depth * 4,
      'depth_bottleneck': base_depth,
      'stride': 1
  }] * (num_units - 1))


def hopenet(inputs, is_training=True, reuse=None, scope='UtilityModule'):
    arg_scope = hopenet_utils.resnet_arg_scope(weight_decay=0.0)
    with slim.arg_scope(arg_scope):
        blocks = [
            resnet_v1_block('block1', base_depth=64, num_units=3, stride=1),
            resnet_v1_block('block2', base_depth=128, num_units=4, stride=2),
            resnet_v1_block('block3', base_depth=256, num_units=6, stride=2),
            resnet_v1_block('block4', base_depth=512, num_units=3, stride=2),
        ]
        with tf.variable_scope(scope, 'resnet_v1', [inputs], reuse=reuse) as sc:
            with slim.arg_scope([slim.conv2d, bottleneck, hopenet_utils.stack_blocks_dense]):
                with slim.arg_scope([slim.batch_norm], is_training=is_training):
                    net = inputs
                    net = hopenet_utils.conv2d_same(net, 64, 7, stride=2, scope='conv1')
                    net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1')
                    print(net.get_shape())

                    net = hopenet_utils.stack_blocks_dense(net, blocks)
                    print(net.get_shape())

                    net = slim.avg_pool2d(net, 7)
                    print(net.get_shape())

                    net = tf.reshape(net, [FLAGS.batch_size, -1])

                    print(net.get_shape())
                    fc_yaw = slim.fully_connected(net, 66, scope='fc_yaw')
                    fc_pitch = slim.fully_connected(net, 66, scope='fc_pitch')
                    fc_roll = slim.fully_connected(net, 66, scope='fc_roll')

                    return fc_yaw, fc_pitch, fc_roll
hopenet.default_image_size = 224