import datetime, os
from argparse import Namespace

# home dir:
home_dir = '/mnt/ilcompf5d1/user/zwu/Privacy-AdversarialLearning/SBU/'

# data dir:
TRAIN_FILES_DIR = os.path.join(home_dir, 'data/adversarial_training/tfrecords/train')
VAL_FILES_DIR = os.path.join(home_dir, 'data/adversarial_training/tfrecords/val')
TEST_FILES_DIR = os.path.join(home_dir, 'data/adversarial_training/tfrecords/test')
FILES_NPY_DIR = os.path.join(home_dir, 'data/adversarial_training/npy')
TRAIN_FILES_DEG_DIR = os.path.join(home_dir, 'data/pretrain_fd/train')
VAL_FILES_DEG_DIR = os.path.join(home_dir, 'data/pretrain_fd/val')

# pretrained fT, fd and fb models:
pretrain_dir = os.path.join(home_dir, 'adversarial_training/pretraining/pretrained')
# pretrained C3D model file path:
PRETRAINED_C3D = os.path.join(pretrain_dir, 'downloaded/C3D/conv3d_deepnetA_sport1m_iter_1900000_TF.model')

# construct namespace object:
COMMON_FLAGS = Namespace()

COMMON_FLAGS.hdd_dir = home_dir

COMMON_FLAGS.TRAIN_FILES_DIR = TRAIN_FILES_DIR # 
COMMON_FLAGS.VAL_FILES_DIR = VAL_FILES_DIR # 
COMMON_FLAGS.TEST_FILES_DIR = TEST_FILES_DIR # 
COMMON_FLAGS.FILES_NPY_DIR = FILES_NPY_DIR

COMMON_FLAGS.TRAIN_FILES_DEG_DIR = TRAIN_FILES_DEG_DIR #
COMMON_FLAGS.VAL_FILES_DEG_DIR = VAL_FILES_DEG_DIR # 

COMMON_FLAGS.pretrain_dir = pretrain_dir # Directory to store pretrain ft fd fb model checkpoints
COMMON_FLAGS.PRETRAINED_C3D = PRETRAINED_C3D # path of pretrained C3D model

# Move DATA and DATA_PROCESSING sections and NUM_EXAMPLES_PER_EPOCH under train section from yaml to common flags:
COMMON_FLAGS.NUM_CLASSES_UTILITY = 8 # 
COMMON_FLAGS.NUM_CLASSES_BUDGET = 13 # 
COMMON_FLAGS.NCHANNEL = 3 # 
COMMON_FLAGS.DEPTH = 16 # 
COMMON_FLAGS.WIDTH = 160 # 
COMMON_FLAGS.HEIGHT = 120 # 
COMMON_FLAGS.CROP_HEIGHT = 112 # 
COMMON_FLAGS.CROP_WIDTH = 112 # 
COMMON_FLAGS.NUM_THREADS = 10 # 

COMMON_FLAGS.NUM_EXAMPLES_PER_EPOCH = 500

COMMON_FLAGS.VAL_NUM = 105 # size of validation set
COMMON_FLAGS.TEST_NUM = 110 # size of testing set