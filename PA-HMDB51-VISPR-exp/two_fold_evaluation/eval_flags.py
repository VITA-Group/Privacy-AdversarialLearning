import argparse, sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))
from common_flags import COMMON_FLAGS
import os

_lambda = 0.5
M = 1
restarting = False
isRestarting = lambda bool: "Restarting_" if bool else "NoRestarting_"
ckpt_dir = os.path.join('adv_ckpts', isRestarting(restarting) + '{}_'.format(M) + '{}'.format(_lambda))


parser = argparse.ArgumentParser()
# GPU section:
parser.add_argument('--GPU_id', default="0,1,2,3")


# Basic model parameters as external flags.
parser.add_argument('--ckpt_dir', type=str, default=os.path.join(COMMON_FLAGS.hdd_dir, 'adversarial_training', 'adv_training', ckpt_dir), help='Directory where to read/write model checkpoints')

# training hyperparameters
parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')
parser.add_argument('--val_step', type=int, default=20, help='Number of steps for validation')
parser.add_argument('--save_step', type=int, default=5, help='Number of step to save the model')
parser.add_argument('--max_steps', type=int, default=1500, help='Number of steps to run trainer.')
parser.add_argument('--weight_decay', type=float, default=0.002, help='The weight decay on the model weights.')

parser.add_argument('--n_minibatches', type=int, default=1, help='Number of mini-batches')

FLAGS = parser.parse_args()
FLAGS.GPU_NUM = int((len(FLAGS.GPU_id)+1)/2)