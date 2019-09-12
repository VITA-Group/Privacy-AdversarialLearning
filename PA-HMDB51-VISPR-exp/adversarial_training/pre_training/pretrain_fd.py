from __future__ import print_function
import sys, os
from os.path import dirname, realpath
sys.path.append(dirname(dirname(dirname(realpath(__file__)))))

from argparse import ArgumentParser
import tensorflow as tf
import numpy as np
import scipy.misc
import functools
import adversarial_training.pre_training.vgg as vgg
from modules.degradNet import fd

from common_flags import COMMON_FLAGS

CONTENT_WEIGHT = 7.5e0
TV_WEIGHT = 2e2

LEARNING_RATE = 1e-3
NUM_EPOCHS = 2
VGG_PATH = 'vgg_pretrained/imagenet-vgg-verydeep-19.mat'
BATCH_SIZE = 4
CONTENT_LAYER = 'relu4_2'
CKPT_DIR = 'pretrained_ckpts/pretrained_fd'
HMDB_DIR = COMMON_FLAGS.HMDB_FRAMES_DIR

def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str,
                        dest='checkpoint_dir', help='dir to save checkpoint in',
                        metavar='CHECKPOINT_DIR', default=CKPT_DIR)

    parser.add_argument('--train_path', type=str,
                        dest='train_path', help='path to training images folder',
                        metavar='TRAIN_PATH', default=HMDB_DIR)

    parser.add_argument('--epochs', type=int,
                        dest='epochs', help='num epochs',
                        metavar='EPOCHS', default=NUM_EPOCHS)

    parser.add_argument('--batch_size', type=int,
                        dest='batch_size', help='batch size',
                        metavar='BATCH_SIZE', default=BATCH_SIZE)

    parser.add_argument('--vgg_path', type=str,
                        dest='vgg_path',
                        help='path to VGG19 network (default %(default)s)',
                        metavar='VGG_PATH', default=VGG_PATH)

    parser.add_argument('--content_weight', type=float,
                        dest='content_weight',
                        help='content weight (default %(default)s)',
                        metavar='CONTENT_WEIGHT', default=CONTENT_WEIGHT)

    parser.add_argument('--tv_weight', type=float,
                        dest='tv_weight',
                        help='total variation regularization weight (default %(default)s)',
                        metavar='TV_WEIGHT', default=TV_WEIGHT)

    parser.add_argument('--learning_rate', type=float,
                        dest='learning_rate',
                        help='learning rate (default %(default)s)',
                        metavar='LEARNING_RATE', default=LEARNING_RATE)

    return parser


def check_opts(opts):
    def exists(p, msg):
        assert os.path.exists(p), msg
    if not os.path.exists(opts.checkpoint_dir):
        os.makedirs(opts.checkpoint_dir)
    exists(opts.train_path, "train path not found!")
    exists(opts.vgg_path, "vgg network data not found!")
    assert opts.epochs > 0
    assert opts.batch_size > 0
    assert os.path.exists(opts.vgg_path)
    assert opts.content_weight >= 0
    assert opts.tv_weight >= 0
    assert opts.learning_rate >= 0


def _get_files(img_dir):
    files = []
    for (dirpath, dirnames, filenames) in os.walk(img_dir):
        files.extend([os.path.join(dirpath, x) for x in filenames])
    return files

def get_img(src, img_size=False):
   img = scipy.misc.imread(src, mode='RGB') # misc.imresize(, (256, 256, 3))
   if not (len(img.shape) == 3 and img.shape[2] == 3):
       img = np.dstack((img,img,img))
   if img_size != False:
       img = scipy.misc.imresize(img, img_size)
   return img

def _tensor_size(tensor):
    from operator import mul
    return functools.reduce(mul, (d.value for d in tensor.get_shape()[1:]), 1)

def main():
    parser = build_parser()
    options = parser.parse_args()
    check_opts(options)

    content_targets = _get_files(options.train_path)

    mod = len(content_targets) % options.batch_size
    if mod > 0:
        print("Train set has been trimmed slightly..")
        content_targets = content_targets[:-mod]

    batch_shape = (options.batch_size,256,256,3)

    with tf.Graph().as_default(), tf.Session() as sess:
        X_content = tf.placeholder(tf.float32, shape=batch_shape, name="X_content")
        X_pre = vgg.preprocess(X_content)

        # precompute content features
        content_features = {}
        content_net = vgg.net(options.vgg_path, X_pre)
        content_features[CONTENT_LAYER] = content_net[CONTENT_LAYER]

        preds = fd(X_content/255.0)
        preds_pre = vgg.preprocess(preds)

        net = vgg.net(options.vgg_path, preds_pre)

        content_size = _tensor_size(content_features[CONTENT_LAYER])*options.batch_size
        assert _tensor_size(content_features[CONTENT_LAYER]) == _tensor_size(net[CONTENT_LAYER])
        content_loss = options.content_weight * (2 * tf.nn.l2_loss(
            net[CONTENT_LAYER] - content_features[CONTENT_LAYER]) / content_size
        )

        # total variation denoising
        tv_y_size = _tensor_size(preds[:,1:,:,:])
        tv_x_size = _tensor_size(preds[:,:,1:,:])
        y_tv = tf.nn.l2_loss(preds[:,1:,:,:] - preds[:,:batch_shape[1]-1,:,:])
        x_tv = tf.nn.l2_loss(preds[:,:,1:,:] - preds[:,:,:batch_shape[2]-1,:])
        tv_loss = options.tv_weight*2*(x_tv/tv_x_size + y_tv/tv_y_size)/options.batch_size

        loss = content_loss + tv_loss

        # overall loss
        train_step = tf.train.AdamOptimizer(options.learning_rate).minimize(loss)
        sess.run(tf.global_variables_initializer())

        for epoch in range(options.epochs):
            num_examples = len(content_targets)
            iterations = 0
            while iterations * options.batch_size < num_examples:
                curr = iterations * options.batch_size
                step = curr + options.batch_size
                X_batch = np.zeros(batch_shape, dtype=np.float32)
                for j, img_p in enumerate(content_targets[curr:step]):
                   X_batch[j] = get_img(img_p, (256,256,3)).astype(np.float32)
                iterations += 1
                assert X_batch.shape[0] == options.batch_size
                train_step.run(feed_dict={X_content:X_batch})
                is_print_iter = int(iterations) % 100 == 0
                is_last = epoch == options.epochs - 1 and iterations * options.batch_size >= num_examples
                should_print = is_print_iter or is_last
                if should_print:
                    _content_loss, _tv_loss, _loss, _preds = sess.run([content_loss, tv_loss, loss, preds], feed_dict = {X_content:X_batch})
                    saver = tf.train.Saver()
                    saver.save(sess, os.path.join(options.checkpoint_dir, 'fns.ckpt'))
                    print('Epoch %d, Iteration: %d, Loss: %s' % (epoch, iterations, _loss))
                    print('content:%s, tv: %s' % (_content_loss, _tv_loss))


if __name__ == '__main__':
    main()
