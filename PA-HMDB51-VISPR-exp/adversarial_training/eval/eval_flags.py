import argparse, sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(dirname(realpath(__file__)))))

from common_flags import COMMON_FLAGS

parser = argparse.ArgumentParser()
# Basic model parameters as external flags.
# GPU section:
parser.add_argument('--GPU_id', default="0,1,2,3")
parser.add_argument('--M', type=int, default=4, help='Number of fbs for ensembling')
parser.add_argument('--weight_decay', type=float, default=0.002, help='The weight decay on the model weights.')
parser.add_argument('--video_batch_size', type=int, default=8, help='Video batch size')
parser.add_argument('--image_batch_size', type=int, default=128, help='Image batch size')
parser.add_argument('--adv_ckpt_dir', type=str, default='adv_ckpts', help='saved ckpts for adversarial trained models')

FLAGS = parser.parse_args()

FLAGS.GPU_NUM = int((len(FLAGS.GPU_id)+1)/2)