import tensorflow as tf
import datetime
import os

flags = tf.app.flags


_gamma = 2.0
_lambda = 0.5
NBudget = 1

# Here resampling means restarting
restarting = True
l1_loss = True
avg_replicate = True
monitor_budget = False
monitor_utility = False
lambda_decay = True

isRestarting = lambda bool: "Restart" if bool else "NoRestart"
isL1Loss = lambda bool: "L1Loss" if bool else "NoL1Loss"
isAvgReplicate = lambda bool: "AvgReplicate" if bool else "NoAvgReplicate"
isMonitorBudget = lambda bool: "MonitorBudget" if bool else "NoMonitorBudget"
isMonitorUtility = lambda bool: "MonitorUtility" if bool else "NoMonitorUtility"
isLambdaDecay = lambda bool: "LambdaDecay" if bool else "NoLambdaDecay"

dir_name = '{}-{}-{}-{}-{}-{}-{}-{}-{}'.format(isRestarting(restarting), NBudget, _gamma, _lambda, isL1Loss(l1_loss),
            isLambdaDecay(lambda_decay), isAvgReplicate(avg_replicate), isMonitorBudget(monitor_budget), isMonitorUtility(monitor_utility))

summary_dir = os.path.join('summaries', dir_name + datetime.datetime.now().strftime("-%Y%m%d_%H%M%S"))
ckpt_dir = os.path.join('ckpt_dir', dir_name)
vis_dir = os.path.join('visualization', dir_name)
log_dir = os.path.join('tensorboard_events', dir_name + datetime.datetime.now().strftime("-%Y%m%d_%H%M%S"))

# Basic model parameters as external flags.
flags.DEFINE_string('summary_dir', summary_dir, 'Directory where to write the summary')
flags.DEFINE_string('ckpt_dir', ckpt_dir, 'Directory where to read/write model checkpoints')
flags.DEFINE_string('vis_dir', vis_dir, 'Directory where to write degradation visualization')
flags.DEFINE_string('log_dir', log_dir, 'Directory where to write the tensorboard events')

flags.DEFINE_string('degradation_models', 'checkpoint/degradation', 'Directory where to read/write model checkpoints')
flags.DEFINE_string('whole_pretraining', 'checkpoint/whole_pretraining', 'Directory where to read/write model checkpoints')

flags.DEFINE_float('_gamma', _gamma, 'Hyperparameter for the weighted combination of utility loss and budget loss')
flags.DEFINE_float('_lambda', _lambda, 'Hyperparameter for the weight of L1 loss')

flags.DEFINE_float('degradation_lr', 1e-4, 'Learning rate for the degradation model')
flags.DEFINE_float('utility_lr', 1e-5, 'Learning rate for the utility model')
flags.DEFINE_float('budget_lr', 1e-2, 'Learning rate for the budget model')

flags.DEFINE_float('highest_util_acc_val', 0.85, 'Monitoring the validation accuracy of the utility task')
flags.DEFINE_float('highest_budget_acc_train', 0.99, 'Monitoring the training accuracy of the budget task')

flags.DEFINE_integer('restarting_step', 200, 'Number of steps for restart model')
flags.DEFINE_integer('retraining_step', 1000, 'Number of steps for retraining the restarted model')

flags.DEFINE_integer('NBudget', NBudget, 'Number of budget models')

flags.DEFINE_boolean('use_xentropy_uniform', True, 'Whether to use cross entropy')

flags.DEFINE_boolean('use_restarting', restarting, 'Whether to use resampling model')

flags.DEFINE_boolean('use_crop', True, 'Whether to use crop when reading video in the input pipeline')
flags.DEFINE_boolean('use_random_crop', False, 'Whether to use random crop when reading video in the input pipeline')
flags.DEFINE_boolean('use_center_crop', True, 'Whether to use center crop when reading video in the input pipeline')
flags.DEFINE_boolean('use_avg_replicate', avg_replicate, 'Whether to replicate the 1 channel into 3 copies after averaging the 3 channels degradation module output')
flags.DEFINE_boolean('use_l1_loss', l1_loss, 'Whether to use the l1 loss regularizer')
flags.DEFINE_boolean('use_monitor_budget', monitor_budget, 'Whether to monitor the budget task')
flags.DEFINE_boolean('use_monitor_utility', monitor_utility, 'Whether to monitor the utility task')
flags.DEFINE_boolean('use_lambda_decay', lambda_decay, 'Whether to use lambda decay')

flags.DEFINE_integer('n_minibatches', 20, 'Number of mini-batches')
flags.DEFINE_integer('n_minibatches_eval', 40, 'Number of mini-batches')

FLAGS = flags.FLAGS