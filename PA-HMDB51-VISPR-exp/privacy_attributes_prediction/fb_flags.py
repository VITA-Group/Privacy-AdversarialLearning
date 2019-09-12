import argparse, sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))

from common_flags import COMMON_FLAGS
import os

parser = argparse.ArgumentParser()

# GPU section:
parser.add_argument('--GPU_id', type=str, default="0")
# trained models
parser.add_argument('--ckpt_dir', type=str, default='ckpts', help='Directory where to read/write model checkpoints')
parser.add_argument('--module_name', type=str, default='resnet_v2_101', help='module name')

# batch size section
parser.add_argument('--batch_size', type=int, default=64, help='Video batch size')

# training hyperparameters
parser.add_argument('--max_steps', type=int, default=500, help='Number of steps to run trainer.')
parser.add_argument('--val_step', type=int, default=25, help='Number of steps for validation')
parser.add_argument('--save_step', type=int, default=25, help='Number of step to save the model')
parser.add_argument('--weight_decay', type=float, default=0.002, help='The weight decay on the model weights.')
parser.add_argument('--factor', type=int, default=1, help='factor for bilinear resizing')

FLAGS = parser.parse_args()

FLAGS.GPU_NUM = int((len(FLAGS.GPU_id)+1)/2)
FLAGS.n_minibatches = int(8/FLAGS.GPU_NUM) # Number of mini-batches
FLAGS.pretrained_ckpt_path = os.path.join(COMMON_FLAGS.hdd_dir, 'model_zoo', COMMON_FLAGS.ckpt_path_map[FLAGS.module_name])