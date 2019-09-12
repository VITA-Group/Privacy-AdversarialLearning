import tensorflow as tf
import datetime

flags = tf.app.flags

_gamma = 500.0
_lambda = 0.5
_mode = 'SuppressingMostConfident'
N = 4

restarting = True
l1_loss = False
monitor_budget = True
monitor_utility = False
lambda_decay = True
residual = True
avg_replicate = True


module_name = 'hopenet'
isRestarting = lambda bool: "Restart" if bool else "NoRestart"
isL1Loss = lambda bool: "L1Loss" if bool else "NoL1Loss"
isAvgReplicate = lambda bool: "AvgReplicate" if bool else "NoAvgReplicate"
isMonitorBudget = lambda bool: "MonitorBudget" if bool else "NoMonitorBudget"
isMonitorUtility = lambda bool: "MonitorUtility" if bool else "NoMonitorUtility"
isLambdaDecay = lambda bool: "LambdaDecay" if bool else "NoLambdaDecay"
isResidual = lambda bool: "UseResidual" if bool else "NoUseResidual"

summary_dir = 'summaries/'  + isL1Loss(l1_loss) + isLambdaDecay(lambda_decay) + isAvgReplicate(avg_replicate) + isMonitorBudget(monitor_budget) + isMonitorUtility(monitor_utility) + isRestarting(restarting) + '{}_'.format(N) + '{}_'.format(_gamma) + '{}_'.format(_lambda) + _mode + '_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"

flags.DEFINE_boolean('use_l1_loss', l1_loss, 'Whether to use the l1 loss regularizer')
flags.DEFINE_boolean('use_avg_replicate', avg_replicate, 'Whether to replicate the 1 channel by averaging the 3 channels degradation module output')
flags.DEFINE_boolean('use_restarting', restarting, 'Whether to use restarting model')
flags.DEFINE_boolean('use_lambda_decay', lambda_decay, 'Whether to use lambda decay')
flags.DEFINE_boolean('use_monitor_budget', monitor_budget, 'Whether to monitor the budget task')
flags.DEFINE_boolean('use_monitor_utility', monitor_utility, 'Whether to monitor the utility task')
flags.DEFINE_boolean('use_residual', residual, 'Whether to use the residual net')

flags.DEFINE_string('summary_dir', summary_dir, 'Directory where to write the summary')
flags.DEFINE_string('mode', 'SuppressingMostConfident', 'Training mode when updating the filtering model')
flags.DEFINE_string('checkpoint_dir', 'checkpoint/adversarial_training', 'Directory where to read/write model checkpoints')
flags.DEFINE_string('degradation_models', 'checkpoint/degradation/{}'.format(isResidual(True)), 'Directory where to read/write model checkpoints')
flags.DEFINE_string('utility_models', 'checkpoint/utility', 'Directory where to read/write model checkpoints')
flags.DEFINE_string('whole_pretraining', 'checkpoint/whole_pretraining/{}'.format(isResidual(True)), 'Directory where to read/write model checkpoints')
flags.DEFINE_string('pretrained_hopenet', 'checkpoint/pretrained_utility', 'Directory of pretrained model checkpoints')
flags.DEFINE_string('train_files_dir', '/home/wuzhenyu_sjtu/DAN_AFLW/gen_tfrecords/tfrecords/train', 'The directory of the training files')
flags.DEFINE_string('val_files_dir', '/home/wuzhenyu_sjtu/DAN_AFLW/gen_tfrecords/tfrecords/val', 'The directory of the training files')

flags.DEFINE_integer('NBudget', N, 'Number of budget models')
flags.DEFINE_integer('pretraining_steps', 500, 'Number of steps to run trainer.')
flags.DEFINE_integer('max_steps', 50000, 'Number of steps to run trainer.')
flags.DEFINE_integer('restart_step', 200, 'Number of steps for restarting model')
flags.DEFINE_integer('retraining_step', 500, 'Number of steps for retraining sampled model')
flags.DEFINE_integer('val_step', 25, 'Number of steps for validation')
flags.DEFINE_integer('save_step', 50, 'Number of step to save the model')
flags.DEFINE_integer('gpu_num', 1, 'Number of gpus to run')
flags.DEFINE_integer('num_threads', 10, 'Number of threads enqueuing tensor list')
flags.DEFINE_integer('num_examples_per_epoch', 15000, 'Epoch size')
flags.DEFINE_integer('n_minibatches', 2, 'Number of mini-batches')
flags.DEFINE_integer('num_classes_budget', 2, 'Number of classes to do classification')
flags.DEFINE_integer('batch_size', 1, 'The number of samples in each batch.')

flags.DEFINE_float('_gamma', _gamma, 'Hyperparameter for the weighted combination of utility loss and budget loss')
flags.DEFINE_float('_lambda', _lambda, 'Hyperparameter for the weight of L1 loss')
flags.DEFINE_float('alpha', 1, 'Regression loss coefficient.')
flags.DEFINE_float('util_loss_val_thresh', 10.0, 'Threshold for the validation utility loss')
flags.DEFINE_float('budget_acc_train_thresh', 0.95, 'Threshold for the validation utility loss')
flags.DEFINE_float('degradation_lr', 1e-4, 'Learning rate for the degradation module')
flags.DEFINE_float('utility_lr', 1e-4, 'Learning rate for the utility module')
flags.DEFINE_float('budget_lr', 1e-3, 'Learning rate for the budget module')


######################
# Optimization Flags #
######################

tf.app.flags.DEFINE_float(
    'weight_decay', 0.00004, 'The weight decay on the model weights.')


tf.app.flags.DEFINE_string(
    'model_name', 'resnet_v1_50', 'The name of the architecture to train.')



#####################
# Fine-Tuning Flags #
#####################

tf.app.flags.DEFINE_string(
    'checkpoint_path', 'resnet_v1_50/resnet_v1_50.ckpt',
    'The path to a checkpoint from which to fine-tune.')


FLAGS = tf.app.flags.FLAGS