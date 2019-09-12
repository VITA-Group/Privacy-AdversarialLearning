import argparse
from common_flags import COMMON_FLAGS

parser = argparse.ArgumentParser()
# bas_idx:
parser.add_argument('--base_idx', type=int, default=0, help='train _K model from k=base_idx')
# GPU section:
parser.add_argument('--GPU_id', type=str, default="6,7", help="")
# lr section:
parser.add_argument('--degradation_lr', type=float, default=1e-4, help='Learning rate for the degradation model')
parser.add_argument('--utility_lr', type=float, default=1e-5, help='Learning rate for the utility model')
parser.add_argument('--budget_lr', type=float, default=1e-2, help='Learning rate for the budget model')
# some other booleans:
parser.add_argument('--use_avg_replicate', type=bool, default=True, help='Whether to replicate the 1 channel into 3 copies after averaging the 3 channels degradation module output')

FLAGS = parser.parse_args()

FLAGS.GPU_NUM = int((len(FLAGS.GPU_id)+1)/2)
FLAGS.n_minibatches = int(80/FLAGS.GPU_NUM) # Number of mini-batches