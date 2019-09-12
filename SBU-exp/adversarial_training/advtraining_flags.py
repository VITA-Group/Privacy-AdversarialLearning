import argparse
from common_flags import COMMON_FLAGS

parser = argparse.ArgumentParser()
# Basic model parameters as external flags.
# GPU section:
parser.add_argument('--GPU_id', default="6,7")
parser.add_argument('--_K', type=int)
# lr section:
parser.add_argument('--degradation_lr', type=float, default=1e-4, help='Learning rate for the degradation model')
parser.add_argument('--utility_lr', type=float, default=1e-5, help='Learning rate for the utility model')
parser.add_argument('--budget_lr', type=float, default=1e-2, help='Learning rate for the budget model')
# monitoring section:
parser.add_argument('--highest_util_acc_val', type=float, default=0.85, help='Monitoring the validation accuracy of the utility task')
parser.add_argument('--highest_budget_acc_val', type=float, default=0.99, help='Monitoring the training accuracy of the budget task')
# restarting section:
parser.add_argument('--restarting_step', type=int, default=200, help='Restart model after this number of steps in the top loop.')
# some other booleans:
parser.add_argument('--use_budget_restarting', type=bool, default=False, help='Whether to use resampling model')
parser.add_argument('--use_avg_replicate', type=bool, default=True, help='Whether to replicate the 1 channel into 3 copies after averaging the 3 channels degradation module output')

FLAGS = parser.parse_args()

FLAGS.GPU_NUM = int((len(FLAGS.GPU_id)+1)/2)
FLAGS.n_minibatches = int(80/FLAGS.GPU_NUM) # Number of mini-batches