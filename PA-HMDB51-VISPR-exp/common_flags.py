from argparse import Namespace
import os

# The home directory of the project
home_dir = '/mnt/ilcompf5d1/user/zwu/Privacy-AdversarialLearning/PA-HMDB51-VISPR/'

# Change the data directory if you are using UCF101
TRAIN_VIDEOS_DIR = os.path.join(home_dir, 'data/adversarial_training/HMDB51/train')
TEST_VIDEOS_DIR = os.path.join(home_dir, 'data/adversarial_training/HMDB51/test')
TRAIN_IMAGES_DIR = os.path.join(home_dir, 'data/adversarial_training/VISPR/5attributes/train')
VAL_IMAGES_DIR = os.path.join(home_dir, 'data/adversarial_training/VISPR/5attributes/val')
TEST_IMAGES_DIR = os.path.join(home_dir, 'data/adversarial_training/VISPR/5attributes/test')
TEST_PA_HMDB51_IMAGES_DIR = os.path.join(home_dir, 'data/adversarial_training/PA_HMDB51/5attributes/test')
HMDB_FRAMES_DIR = os.path.join(home_dir, 'data/pretrain_fd/HMDB51_frames')

PRETRAINED_C3D_DIR = os.path.join(home_dir, 'pretrained/C3D/conv3d_deepnetA_sport1m_iter_1900000_TF.model')

COMMON_FLAGS = Namespace()

COMMON_FLAGS.hdd_dir = home_dir

COMMON_FLAGS.TRAIN_VIDEOS_DIR = TRAIN_VIDEOS_DIR
COMMON_FLAGS.TEST_VIDEOS_DIR = TEST_VIDEOS_DIR
COMMON_FLAGS.TRAIN_IMAGES_DIR = TRAIN_IMAGES_DIR
COMMON_FLAGS.VAL_IMAGES_DIR = VAL_IMAGES_DIR
COMMON_FLAGS.TEST_IMAGES_DIR = TEST_IMAGES_DIR
COMMON_FLAGS.TEST_PA_HMDB51_IMAGES_DIR = TEST_PA_HMDB51_IMAGES_DIR
COMMON_FLAGS.HMDB_FRAMES_DIR = HMDB_FRAMES_DIR

COMMON_FLAGS.PRETRAINED_C3D_DIR = PRETRAINED_C3D_DIR

# Change to 101 if you are using UCF101
COMMON_FLAGS.NUM_CLASSES_UTILITY = 51 #
# Skin color, face, gender, nudity, relationship
COMMON_FLAGS.NUM_CLASSES_BUDGET = 5 #
COMMON_FLAGS.NCHANNEL = 3 #
COMMON_FLAGS.DEPTH = 16 #
COMMON_FLAGS.WIDTH = 160 #
COMMON_FLAGS.HEIGHT = 120 #
COMMON_FLAGS.CROP_HEIGHT = 112 #
COMMON_FLAGS.CROP_WIDTH = 112 #
COMMON_FLAGS.NUM_THREADS = 10 #
COMMON_FLAGS.NUM_EXAMPLES_PER_EPOCH = 5000


COMMON_FLAGS.ckpt_path_map = {
                'vgg_16':               'vgg_16/vgg_16.ckpt',
                'vgg_19':               'vgg_19/vgg_19.ckpt',
                'inception_v1':         'inception_v1/inception_v1.ckpt',
                'inception_v2':         'inception_v2/inception_v2.ckpt',
                'inception_v3':         'inception_v3/inception_v3.ckpt',
                'inception_v4':         'inception_v4/inception_v4.ckpt',
                'resnet_v1_50':         'resnet_v1_50/resnet_v1_50.ckpt',
                'resnet_v1_101':        'resnet_v1_101/resnet_v1_101.ckpt',
                'resnet_v1_152':        'resnet_v1_152/resnet_v1_152.ckpt',
                'resnet_v2_50':         'resnet_v2_50/resnet_v2_50.ckpt',
                'resnet_v2_101':        'resnet_v2_101/resnet_v2_101.ckpt',
                'resnet_v2_152':        'resnet_v2_152/resnet_v2_152.ckpt',
                'mobilenet_v1':         'mobilenet_v1_1.0_128/',
                'mobilenet_v1_075':     'mobilenet_v1_0.75_128/',
                'mobilenet_v1_050':     'mobilenet_v1_0.50_128/',
                'mobilenet_v1_025':     'mobilenet_v1_0.25_128/',
               }