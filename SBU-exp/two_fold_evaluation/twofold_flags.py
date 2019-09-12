import os, argparse
from common_flags import COMMON_FLAGS

parser = argparse.ArgumentParser()
# GPU section:
parser.add_argument('--GPU_id', default="3,4")
# adversarial_ckpt_file:
parser.add_argument('--adversarial_job_name', default="kbeam-NoRest-K1")

FLAGS = parser.parse_args()

FLAGS.GPU_NUM = int((len(FLAGS.GPU_id)+1)/2)
FLAGS.n_minibatches = int(80/FLAGS.GPU_NUM) # Number of mini-batches

# load pretrained adversarial model from this dir:
FLAGS.adversarial_ckpt_file_dir = os.path.join(COMMON_FLAGS.hdd_dir, 'adversarial_training', FLAGS.adversarial_job_name, 'ckpt_dir')

# save results dir:
# write summary in this dir:
FLAGS.summary_dir = os.path.join(COMMON_FLAGS.hdd_dir, 'two_fold_evaluation', FLAGS.adversarial_job_name, 'summaries')
# save budget model ckpt in this dir:
FLAGS.two_fold_eval_ckpt_dir = os.path.join(COMMON_FLAGS.hdd_dir, 'two_fold_evaluation', FLAGS.adversarial_job_name, 'checkpoint_eval')
# save testing result in this dir:
FLAGS.test_result_dir = os.path.join(COMMON_FLAGS.hdd_dir, 'two_fold_evaluation', FLAGS.adversarial_job_name, 'test_result')