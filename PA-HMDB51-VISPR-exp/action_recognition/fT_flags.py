import argparse

parser = argparse.ArgumentParser()
# GPU section:
parser.add_argument('--GPU_id', type=str, default="0,1", help="")
# lr section:
parser.add_argument('--fT_main_lr', type=float, default=1e-5, help='Learning rate for fT')
parser.add_argument('--fT_finetune_lr', type=float, default=1e-4, help='Learning rate for fT')
# batch size section
parser.add_argument('--video_batch_size', type=int, default=16, help='Video batch size')
parser.add_argument('--weight_decay', type=float, default=0.002, help='The weight decay on the model weights.')
# pre-trained models
parser.add_argument('--ckpt_dir', type=str, default='ckpts', help='saved ckpts for pretrained fd+fT')
# training hyperparameters
parser.add_argument('--training_steps_fT', type=int, default=500, help='Number of steps for pretraining fT+fd')
parser.add_argument('--val_step', type=int, default=25, help='Number of steps for evaluating performance on train set and val set')
parser.add_argument('--save_step', type=int, default=25, help='Number of steps for saving the model')
parser.add_argument('--factor', type=int, default=1, help='factor for bilinear resizing')
FLAGS = parser.parse_args()

FLAGS.GPU_NUM = int((len(FLAGS.GPU_id)+1)/2)
FLAGS.n_minibatches = int(8/FLAGS.GPU_NUM) # Number of mini-batches