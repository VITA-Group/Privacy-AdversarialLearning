import argparse, sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(dirname(realpath(__file__)))))
import os
import datetime

_lambda = 0.5
M = 1
restarting = False
isRestarting = lambda bool: "Restarting_" if bool else "NoRestarting_"
ckpt_dir = os.path.join('adv_ckpts', isRestarting(restarting) + '{}_'.format(M) + '{}'.format(_lambda), datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
summary_dir = os.path.join('summaries', isRestarting(restarting) + '{}_'.format(M) + '{}'.format(_lambda), datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))

parser = argparse.ArgumentParser()
# Basic model parameters as external flags.
# Number of fbs for ensembling
parser.add_argument('--M', type=int, default=M, help='Number of fbs for ensembling')
# GPU section:
parser.add_argument('--GPU_id', type=str, default="0,1")
# Training hyperparameters
parser.add_argument('--gamma', type=float, default=0.5, help='gamma for the weight of adversarial loss')
# lr section:
parser.add_argument('--fd_lr', type=float, default=1e-4, help='Learning rate for fd')
parser.add_argument('--fT_lr', type=float, default=1e-5, help='Learning rate for fT')
parser.add_argument('--fb_lr', type=float, default=1e-2, help='Learning rate for fb')
# monitoring section:
parser.add_argument('--fT_acc_val_thresh', type=float, default=0.70, help='Monitoring the validation accuracy of the target task')
# restarting section:
parser.add_argument('--restarting_step', type=int, default=200, help='Restart model after this number of steps in the top loop.')
# some other booleans:
parser.add_argument('--use_fb_restarting', type=bool, default=False, help='Whether to use resampling model')
parser.add_argument('--use_avg_replicate', type=bool, default=True, help='Whether to replicate the 1 channel into 3 copies after averaging the 3 channels degradation module output')
# batch size section:
parser.add_argument('--video_batch_size', type=int, default=2, help='Video batch size')
parser.add_argument('--image_batch_size', type=int, default=8, help='Image batch size')
parser.add_argument('--weight_decay', type=float, default=0.002, help='The weight decay on the model weights.')
# pre-trained models
parser.add_argument('--pretrained_fd_ckpt_dir', type=str, default='../pre_training/pretrained_ckpts/pretrained_fd', help='saved ckpts for pretrained fd')
parser.add_argument('--pretrained_fdfT_ckpt_dir', type=str, default='../pre_training/pretrained_ckpts/pretrained_fdfT', help='saved ckpts for pretrained fd+fT')
parser.add_argument('--pretrained_fbfdfT_ckpt_dir', type=str, default='../pre_training/pretrained_ckpts/pretrained_fbfdfT', help='saved ckpts for pretrained fb')
# adv-trained models and summaries
parser.add_argument('--adv_ckpt_dir', type=str, default=ckpt_dir, help='saved ckpts for adversarial trained models')
parser.add_argument('--summary_dir', type=str, default=summary_dir, help='summaries keep track of the performance in the adversarial training')
# training hyperparameters
parser.add_argument('--val_step', type=int, default=20, help='Number of steps for evaluating performance on train set and val set')
parser.add_argument('--save_step', type=int, default=20, help='Number of steps for saving the model')
parser.add_argument('--max_steps', type=int, default=800, help='Maximum number of steps for adversarial training')
parser.add_argument('--pretraining_steps_fb', type=int, default=200, help='Number of step for the pretraining')

FLAGS = parser.parse_args()

FLAGS.GPU_NUM = int((len(FLAGS.GPU_id)+1)/2)
FLAGS.n_minibatches = int(80/FLAGS.GPU_NUM) # Number of mini-batches