import argparse

parser = argparse.ArgumentParser()
# bas_idx:
parser.add_argument('--base_idx', type=int, default=0, help='train _K model from k=base_idx')
# GPU section:
parser.add_argument('--GPU_id', type=str, default="6,7", help="")
# lr section:
parser.add_argument('--fd_lr', type=float, default=1e-4, help='Learning rate for fd')
parser.add_argument('--fT_main_lr', type=float, default=1e-5, help='Learning rate for fT')
parser.add_argument('--fT_finetune_lr', type=float, default=1e-4, help='Learning rate for fT')
parser.add_argument('--fb_lr', type=float, default=1e-3, help='Learning rate for fb')
# some other booleans:
parser.add_argument('--use_avg_replicate', type=bool, default=True, help='Whether to replicate the 1 channel into 3 copies after averaging the 3 channels degradation module output')
# batch size section
parser.add_argument('--video_batch_size', type=int, default=2, help='Video batch size')
parser.add_argument('--image_batch_size', type=int, default=8, help='Image batch size')
parser.add_argument('--weight_decay', type=float, default=0.002, help='The weight decay on the model weights.')
# pre-trained models
parser.add_argument('--pretrained_fd_ckpt_dir', type=str, default='pretrained_ckpts/pretrained_fd', help='saved ckpts for pretrained fd')
parser.add_argument('--pretrained_fT_ckpt_dir', type=str, default='pretrained_ckpts/pretrained_fT', help='saved ckpts for pretrained fT')
parser.add_argument('--pretrained_fb_ckpt_dir', type=str, default='pretrained_ckpts/pretrained_fb', help='saved ckpts for pretrained fb')
# training hyperparameters
parser.add_argument('--pretraining_steps_fT', type=int, default=500, help='Number of steps for pretraining fT')
parser.add_argument('--pretraining_steps_fb', type=int, default=1000, help='Number of steps for pretraining fb')
parser.add_argument('--val_step', type=int, default=5, help='Number of steps for evaluating performance on train set and val set')
parser.add_argument('--save_step', type=int, default=10, help='Number of steps for saving the model')
FLAGS = parser.parse_args()

FLAGS.GPU_NUM = int((len(FLAGS.GPU_id)+1)/2)
FLAGS.n_minibatches = int(80/FLAGS.GPU_NUM) # Number of mini-batches