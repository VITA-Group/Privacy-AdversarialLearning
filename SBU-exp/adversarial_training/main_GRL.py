'''
GRL
'''

import sys
sys.path.insert(0, '..')

import time
import datetime
from six.moves import xrange
import input_data
import errno
import pprint
import itertools
from degradlNet import residualNet
from budgetNet import budgetNet
from utilityNet import utilityNet
from loss import *
from utils import *
from img_proc import _avg_replicate
import yaml
from tf_flags import FLAGS

from functions import placeholder_inputs, create_grad_accum_for_late_update, create_videos_reading_ops, create_summary_files
from bcolors import bcolors

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ["CUDA_VISIBLE_DEVICES"]=FLAGS.GPU_id
print('Using GPU:', FLAGS.GPU_id)
print('GPU_NUM:', FLAGS.GPU_NUM)

# job name:
algo_str = 'GRL'
restarting = False
rest_str = 'Rest' if restarting else 'NoRest'

dir_name = '{}-{}-M{}'.format(algo_str, rest_str, FLAGS.NBudget)
if not os.path.isdir(dir_name):
	os.mkdir(dir_name)
# end job name

# dir name:
summary_dir = os.path.join(dir_name , 'summaries' + datetime.datetime.now().strftime("-%Y%m%d_%H%M%S"))
ckpt_dir = os.path.join(dir_name, 'ckpt_dir')
# vis_dir = os.path.join(dir_name, 'visualization')
# log_dir = os.path.join(dir_name, 'tensorboard_events')
test_result_dir = os.path.join(dir_name, 'testing_results')
# end dir names

use_pretrained_model = True # if True, load ckpts from pretrained fd and fT; if False: load ckpts from previous training stage of this run_adversarial_training function.


def create_architecture_adversarial(cfg, batch_size, multiplier_lst, logits_budget_lst_dct, loss_budget_lst_dct, scope, videos, utility_labels, budget_labels, dropout):
	'''
	Create the architecture of the adversarial model in the graph
	is_training: whether it is in the adversarial training part. (include testing, not two-fold)
	'''
	# fd part:
	degrad_videos = residualNet(videos, is_video=True)
	degrad_videos = _avg_replicate(degrad_videos) if FLAGS.use_avg_replicate else degrad_videos
	# fd part ends
	# fT part:
	logits_utility = utilityNet(degrad_videos, dropout, wd=0.001)
	loss_utility = tower_loss_xentropy_sparse(scope, logits_utility, utility_labels, use_weight_decay=True)
	# fT part ends
	# fb part:
	logits_budget = tf.zeros([batch_size, cfg['DATA']['NUM_CLASSES_BUDGET']])
	loss_budget = 0.0
	budget_logits_lst = []
	for multiplier in multiplier_lst:
		print(multiplier)
		logits = budgetNet(degrad_videos, depth_multiplier=multiplier)
		budget_logits_lst.append(logits)
		loss = tower_loss_xentropy_sparse(scope, logits, budget_labels, use_weight_decay=False)
		logits_budget_lst_dct[str(multiplier)].append(logits)
		loss_budget_lst_dct[str(multiplier)].append(loss)
		logits_budget += logits / FLAGS.NBudget
		loss_budget += loss / FLAGS.NBudget
	# fd part ends.
	# Find the largest budget loss of the M ensembled budget models:
	argmax_adverse_budget_loss = None
	# finish finding max_adverse_budget_loss and argmax_adverse_budget_loss.

	# change the definition of loss_degrade:
	loss_degrade = loss_utility - cfg['TRAIN']['_lambda'] * loss_budget
	
	return loss_degrade, loss_budget, loss_utility, logits_budget, logits_utility, argmax_adverse_budget_loss

# Training set for traning, validation set for validation.
# lambda: weight for L1 loss (degrade model L1 loss)
def run_adversarial_training(cfg):
	'''
	Algorithm 1 in the paper
	'''

	def run_validation(input_op_list, summary_file, summary_info):
		'''
		Validation during training.
		Validation can be run on any set: training, validating or testing.

		Input:
		sess: run oprations in this session.
		input_op_list: list. For example, when validating on training set, it is [tr_videos_op, tr_action_labels_op, tr_actor_labels_op]
		other_op_list: list. Always [acc_utility, acc_budget, loss_utility, loss_budget]
		summary_file: put the validation summary in this file.
		summary_info: string. Summary content.

		Output:
		print and write summary.

		Return:
		acc_util_lst, acc_budget_lst
		'''

		# initialize timer and lists:
		start_time = time.time()
		acc_util_lst, acc_budget_lst, loss_utility_lst, loss_budget_lst = [], [], [], []
		# late update:
		for _ in itertools.repeat(None, FLAGS.n_minibatches_eval):
			tr_videos, tr_action_labels, tr_actor_labels = sess.run(input_op_list)
			acc_t, acc_b, loss_t, loss_b = sess.run(
									[acc_utility, acc_budget, loss_utility, loss_budget],
									feed_dict={videos_placeholder: tr_videos,
												utility_labels_placeholder: tr_action_labels,
												budget_labels_placeholder: tr_actor_labels,
												dropout_placeholder: 1.0,
												})
			acc_util_lst.append(acc_t)
			acc_budget_lst.append(acc_b)
			loss_utility_lst.append(loss_t)
			loss_budget_lst.append(loss_b)
		# Writing and printing summary part:
		summary = summary_info.format(
				time.time() - start_time, 
				np.mean(acc_util_lst), np.mean(acc_budget_lst),
				np.mean(loss_utility_lst), np.mean(loss_budget_lst))
		print(summary)
		summary_file.write(summary + '\n')
		print('\n')
		# End writing and printing summary part.
		return acc_util_lst, acc_budget_lst

	# initialize multiplier_lst, logits_budget_lst_dct, loss_budget_lst_dct, which are used in both the graph and the session:
	# The depth multiplier list for creating different budget models ensemble (MobileNet with different depth.)
	multiplier_lst = [0.60 - i * 0.02 for i in range(FLAGS.NBudget)]
	# The dict of logits and loss for each different budget model to get accuracy
	logits_budget_lst_dct = {str(multiplier): [] for multiplier in multiplier_lst}
	loss_budget_lst_dct = {str(multiplier): [] for multiplier in multiplier_lst}
	# end initializing multiplier_lst, logits_budget_lst_dct, loss_budget_lst_dct.

	# mkdir for saving ckpt of the adversarial training process:
	if not os.path.exists(ckpt_dir):
		os.makedirs(ckpt_dir)

	# define graph
	graph = tf.Graph()
	with graph.as_default():
		# global step:
		global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
		# placeholder inputs:
		videos_placeholder, utility_labels_placeholder, budget_labels_placeholder, dropout_placeholder, _ = \
										placeholder_inputs(cfg['TRAIN']['BATCH_SIZE'] * FLAGS.GPU_NUM, cfg)

		tower_grads_degrade, tower_grads_utility, tower_grads_budget = [], [], []

		# Compute Acc (fT, fb logits output)
		logits_utility_lst, logits_budget_lst = [], []

		# Compute Loss (LT, Lb_cross_entropy, Ld=LT+Lb_entropy?)
		loss_utility_lst, loss_budget_lst, loss_degrade_lst = [], [], []

		# Compute prediction with min entropy (most confident)
		# Use max uniform loss instead
		argmax_adverse_budget_loss_lst = []

		# Optimizer for the 3 components respectively
		opt_degrade = tf.train.AdamOptimizer(FLAGS.degradation_lr)
		opt_utility = tf.train.AdamOptimizer(FLAGS.utility_lr)
		opt_budget = tf.train.AdamOptimizer(FLAGS.budget_lr)
		

		with tf.variable_scope(tf.get_variable_scope()):
			for gpu_index in range(0, FLAGS.GPU_NUM):
				with tf.device('/gpu:%d' % gpu_index):
					print('/gpu:%d' % gpu_index)
					with tf.name_scope('%s_%d' % ('gpu', gpu_index)) as scope:
						videos = videos_placeholder[gpu_index * cfg['TRAIN']['BATCH_SIZE']:(gpu_index + 1) * cfg['TRAIN']['BATCH_SIZE']]
						utility_labels = utility_labels_placeholder[gpu_index * cfg['TRAIN']['BATCH_SIZE']:(gpu_index + 1) * cfg['TRAIN']['BATCH_SIZE']]
						budget_labels = budget_labels_placeholder[gpu_index * cfg['TRAIN']['BATCH_SIZE']:(gpu_index + 1) * cfg['TRAIN']['BATCH_SIZE']]
						loss_degrade, loss_budget, loss_utility, logits_budget, logits_utility, argmax_adverse_budget_loss = \
									create_architecture_adversarial(cfg, cfg['TRAIN']['BATCH_SIZE'], multiplier_lst, logits_budget_lst_dct, loss_budget_lst_dct, scope, videos, utility_labels, budget_labels, dropout_placeholder)
						
						# Reuse variables for the next tower.
						tf.get_variable_scope().reuse_variables()
						
				loss_degrade_lst.append(loss_degrade)
				loss_budget_lst.append(loss_budget)
				loss_utility_lst.append(loss_utility)
				logits_budget_lst.append(logits_budget)
				logits_utility_lst.append(logits_utility)
				argmax_adverse_budget_loss_lst.append(argmax_adverse_budget_loss)
				# varlist:
				varlist_degrade = [v for v in tf.trainable_variables() if any(x in v.name for x in ["DegradationModule"])]
				varlist_utility = [v for v in tf.trainable_variables() if any(x in v.name for x in ["UtilityModule"])]
				varlist_budget = [v for v in tf.trainable_variables() if any(x in v.name for x in ["BudgetModule"])]

				grads_degrade = opt_degrade.compute_gradients(loss_degrade, varlist_degrade)
				grads_budget = opt_budget.compute_gradients(loss_budget, varlist_budget)
				grads_utility = opt_utility.compute_gradients(loss_utility, varlist_utility)

				tower_grads_degrade.append(grads_degrade)
				tower_grads_budget.append(grads_budget)
				tower_grads_utility.append(grads_utility)

						
		# argmax_adverse_budget_loss_op = tf.concat(argmax_adverse_budget_loss_lst, 0)

		# Average losses over each GPU:
		loss_utility = tf.reduce_mean(loss_utility_lst, name='softmax') # LT
		loss_budget = tf.reduce_mean(loss_budget_lst, name='softmax') # Lb
		loss_degrade = tf.reduce_mean(loss_degrade_lst, name='softmax') # Ld = LT-Lb

		# Concatenate the logits over all GPU:
		logits_utility = tf.concat(logits_utility_lst, 0)
		logits_budget = tf.concat(logits_budget_lst, 0)
		# acc
		acc_utility = accuracy(logits_utility, utility_labels_placeholder)
		acc_budget = accuracy(logits_budget, budget_labels_placeholder)
		# count how many testing samples are classified correctly:
		right_count_utility_op = correct_num(logits_utility, utility_labels_placeholder)
		right_count_budget_op = correct_num(logits_budget, budget_labels_placeholder)

		# operations on each budget model:
		loss_budget_op_lst = []
		accuracy_budget_list = []
		right_count_budget_op_lst = []
		# for each mobile-net:
		for multiplier in multiplier_lst: # multiplier_lst has M elements -> each is the channel depth of a mobile net.
			# loss of each model:
			loss_budget_op_each_model = tf.reduce_mean(loss_budget_lst_dct[str(multiplier)]) # mean loss over multi-gpu of a certain mobile-net.
			loss_budget_op_lst.append(loss_budget_op_each_model)
			# logits of each model:
			budget_logits_each_model = tf.concat(logits_budget_lst_dct['{}'.format(multiplier)], 0) # same budget model, concatenate over GPUs.
			# acc of each model
			accuracy_budget_each_model = accuracy(budget_logits_each_model, budget_labels_placeholder)
			accuracy_budget_list.append(accuracy_budget_each_model)
			# right count of each model:
			right_count_op = correct_num(budget_logits_each_model, budget_labels_placeholder)
			right_count_budget_op_lst.append(right_count_op)

		zero_ops_degrade, accum_ops_degrade, apply_gradient_op_degrade = create_grad_accum_for_late_update(opt_degrade, tower_grads_degrade, varlist_degrade, global_step, decay_with_global_step=True)
		zero_ops_budget, accum_ops_budget, apply_gradient_op_budget = create_grad_accum_for_late_update(opt_budget, tower_grads_budget, varlist_budget, global_step, decay_with_global_step=False)
		zero_ops_utility, accum_ops_utility, apply_gradient_op_utility = create_grad_accum_for_late_update(opt_utility, tower_grads_utility, varlist_utility, global_step, decay_with_global_step=False)

		tr_videos_op, tr_action_labels_op, tr_actor_labels_op = create_videos_reading_ops(is_train=True, is_val=False, cfg=cfg)
		val_videos_op, val_action_labels_op, val_actor_labels_op = create_videos_reading_ops(is_train=False, is_val=True, cfg=cfg)

	# session config:
	config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
	config.gpu_options.allow_growth = True
	# run session:
	with tf.Session(graph=graph, config=config) as sess:
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess=sess, coord=coord)
		init_op = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())
		sess.run(init_op)

		# load ckpts: 
		if use_pretrained_model: # load ckpts from pretrained fd and fT.(By run_pretraining_degrade and run_pretraining_utility functions.)
			# fT and fd part:
			restore_model_ckpt(sess, FLAGS.deg_target_models, varlist_utility+varlist_degrade) # FLAGS.deg_target_models is the dir storing ckpt of theta_T and theta_d
			# fb part:
			restore_model_ckpt(sess, FLAGS.budget_models, varlist_budget)
		else: # load ckpts from previous training stage of this run_adversarial_training function.
			saver = tf.train.Saver(tf.trainable_variables())
			ckpt = tf.train.get_checkpoint_state(checkpoint_dir=ckpt_dir)
			if ckpt and ckpt.model_checkpoint_path:
				saver.restore(sess, ckpt.model_checkpoint_path)
				print('Session restored from trained model at {}!'.format(ckpt.model_checkpoint_path))
			else:
				raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), ckpt_dir)
		
		# saver and summary files:
		saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=10)
		loss_summary_file, validation_train_set_summary_file, validation_val_set_summary_file, _ = create_summary_files(summary_dir)

		# Adversarial training loop:
		for step in xrange(cfg['TRAIN']['TOP_MAXSTEP']):

			# Part 1: optimize theta_b using L_b (cross entropy)
			for L_b_step in range(0, cfg['TRAIN']['L_B_STEP']):

				start_time = time.time()
				acc_budget_lst, loss_budget_lst = [], []
				sess.run(zero_ops_degrade)
				# accumulating gradient for late update:
				for _ in itertools.repeat(None, FLAGS.n_minibatches):
					# placeholder inputs:
					tr_videos, tr_actor_labels = sess.run([tr_videos_op, tr_actor_labels_op])
					# run operations:
					_, acc_b, loss_b = sess.run([accum_ops_budget, acc_budget, loss_budget],
									feed_dict={
											videos_placeholder: tr_videos,
											budget_labels_placeholder: tr_actor_labels,
											dropout_placeholder: 1.0})
					# append loss and acc for budget model:
					acc_budget_lst.append(acc_b)
					loss_budget_lst.append(loss_b)
				# finish accumulating gradient for late update

				# # Monitoring fb using training set
				# if L_b_step % cfg['TRAIN']['MONITOR_STEP'] == 0:
				# 	if np.mean(acc_budget_lst) >= FLAGS.highest_budget_acc_val:
				# 		print(bcolors.OKGREEN + 'pass budget acc bar!\n' + bcolors.ENDC)
				# 		loss_summary_file.write('pass budget acc bar!\n')
				# 		break
				# # End monitoring fb on training set.

				# after accumulating gradient, do the update on fd:
				_ = sess.run([apply_gradient_op_budget])
				# finish update on fd

				assert not np.isnan(np.mean(loss_b)), 'Model diverged with loss = NaN'

				# loss summary:
				if L_b_step % cfg['TRAIN']['PRINT_STEP'] == 0:
					loss_summary = 'Step: {:4d}, L_b_step: {:4d}, time: {:.4f}, ' \
								'training budget accuracy: {:.5f}, budget loss: {:.8f}'.format(
								step, L_b_step, time.time() - start_time,
								np.mean(acc_budget_lst), np.mean(loss_budget_lst)
								)

					print(loss_summary)
					loss_summary_file.write(loss_summary + '\n')
				# end loss summary
			print()
			# End part 1.


			# Part 2: train Ft using L_T
			for L_T_step in range(0, cfg['TRAIN']['L_T_STEP']):

				# # Monitoring LT using validation set:
				# if L_T_step % cfg['TRAIN']['MONITOR_STEP'] == 0:
				# 	acc_util_lst, _ = run_validation(input_op_list=[val_videos_op, val_action_labels_op, val_actor_labels_op], 
				# 							summary_file=loss_summary_file, 
				# 							summary_info="Monitoring L_T:\n" \
				# 										"Step: %d, L_T_step: %d, time: {:.4f}, " \
				# 										"validation utility accuracy: {:.5f}, validation budget accuracy: {:.5f}, " \
				# 										"utility loss: {:.8f}, budget loss: {:.8f}" % (step, L_T_step))
				# 	# breaking condition: (if performance on L_T is still good)
				# 	if np.mean(acc_util_lst) >= FLAGS.highest_util_acc_val:
				# 		print(bcolors.OKGREEN  + 'pass utility acc bar!\n' + bcolors.ENDC)
				# 		loss_summary_file.write('pass utility acc bar!\n')
				# 		break
				# # End of monitoring LT

				# Optimizing LT (if necessary) using training set: (This is one batch=FLAGS.n_minibatches, each minibatch has FLAGS.GPU_NUM*cfg['TRAIN']['BATCH_SIZE'] video clips.)               
				start_time = time.time()
				sess.run(zero_ops_utility)
				acc_util_lst, loss_utility_lst = [], []
				# accumulating gradient for late update:
				for _ in itertools.repeat(None, FLAGS.n_minibatches):
					tr_videos, tr_action_labels = sess.run([tr_videos_op, tr_action_labels_op])
					_, acc_t, loss_t = sess.run([accum_ops_utility, acc_utility, loss_utility],
								feed_dict={videos_placeholder: tr_videos,
											utility_labels_placeholder: tr_action_labels,
											dropout_placeholder: 0.5})
					acc_util_lst.append(acc_t)
					loss_utility_lst.append(loss_t)
				# finish accumulating gradient for late update
				# after accumulating gradient, do the update on fT and fd:
				sess.run([apply_gradient_op_utility])
				# finish update on fT and fd

				assert not np.isnan(np.mean(loss_utility_lst)), 'Model diverged with loss = NaN'

				# loss summary:
				loss_summary = 'Step: {:4d}, L_T_step: {:4d}, time: {:.4f}, ' \
							'training utility accuracy: {:.5f}, utility loss: {:.8f}'.format(
							step, L_T_step, time.time() - start_time,
							np.mean(acc_util_lst), np.mean(loss_utility_lst)
							)

				print(loss_summary)
				loss_summary_file.write(loss_summary + '\n')
				# end of loss summary
				# End of optimizing LT.
			
			print('')
			loss_summary_file.write('\n')
			# End part 2


			# part 3: optimize theta_d by minimizing L_T(X,Y_T)-L_b(X,Y_B)
			for L_d_step in range(0, cfg['TRAIN']['L_D_STEP']):

				start_time = time.time()
				sess.run(zero_ops_budget)
				acc_util_lst, loss_utility_lst, acc_budget_lst, loss_budget_lst, loss_degrade_lst = [], [], [], [], []
				# accumulating gradient for late update:
				for _ in itertools.repeat(None, FLAGS.n_minibatches):
					# placeholder inputs:
					tr_videos, tr_action_labels, tr_actor_labels = sess.run([tr_videos_op, tr_action_labels_op, tr_actor_labels_op])
					# run operations:
					_, acc_b, loss_b, acc_t, loss_t, loss_d = sess.run(
								[accum_ops_degrade, acc_budget, loss_budget, acc_utility, loss_utility, loss_degrade],
								feed_dict={
										videos_placeholder: tr_videos,
										utility_labels_placeholder: tr_action_labels,
										budget_labels_placeholder: tr_actor_labels,
										dropout_placeholder: 1.0})
					# append loss and acc for budget model:
					acc_budget_lst.append(acc_b)
					loss_budget_lst.append(loss_b)
					acc_util_lst.append(acc_t)
					loss_utility_lst.append(loss_t)
					loss_degrade_lst.append(loss_d)
				# finish accumulating gradient for late update

				assert not np.isnan(np.mean(loss_budget_lst)), 'Model diverged with loss = NaN'

				# after accumulating gradient, do the update on fb, if it didn't pass the budget acc bar:
				sess.run([apply_gradient_op_degrade])
				# finish update on fb

				# loss summary:
				if L_d_step % cfg['TRAIN']['PRINT_STEP'] == 0:
					loss_summary = 'Step: {:4d}, L_d_step: {:4d}, time: {:.4f}, ' \
								'training budget accuracy: {:.5f}, budget loss: {:.8f}, ' \
								'training utility accuracy: {:.5f}, utility loss: {:.8f}, ' \
								'degrade loss: {:.8f}'.format(
								step, L_d_step, time.time() - start_time,
								np.mean(acc_budget_lst), np.mean(loss_budget_lst),
								np.mean(acc_util_lst), np.mean(loss_utility_lst),
								np.mean(loss_degrade_lst)
								)

					print(loss_summary)
					loss_summary_file.write(loss_summary + '\n')
				# end loss summary
				
			print('')
			loss_summary_file.write('\n')
			# End part 3

			# Do validation (on training set and validation set):
			if step % cfg['TRAIN']['VAL_STEP'] == 0:
				# run_validation(input_op_list=[tr_videos_op, tr_action_labels_op, tr_actor_labels_op], 
				# 			summary_file=validation_train_set_summary_file, 
				# 			summary_info="Validation train_set summary\n" \
				# 						"Step: %d, time: {:.4f}, " \
				# 						"training utility accuracy: {:.5f}, training budget accuracy: {:.5f}, " \
				# 						"utility loss: {:.8f}, budget loss: {:.8f}" % step)
				run_validation(input_op_list=[val_videos_op, val_action_labels_op, val_actor_labels_op], 
							summary_file=validation_val_set_summary_file, 
							summary_info="Validation val_set summary\n" \
										"Step: %d, time: {:.4f}, " \
										"validation utility accuracy: {:.5f}, validation budget accuracy: {:.5f}, " \
										"utility loss: {:.8f}, budget loss: {:.8f}" % step)
			# End evaluation

			# Save ckpt for kb_adversarial learning:
			if step % cfg['TRAIN']['SAVE_STEP'] == 0 or (step + 1) == cfg['TRAIN']['TOP_MAXSTEP']:
				checkpoint_path = os.path.join(ckpt_dir, 'model.ckpt')
				saver.save(sess, checkpoint_path, global_step=step)
			# End evaluation

		loss_summary_file.close()
		validation_train_set_summary_file.close()
		validation_val_set_summary_file.close()
		coord.request_stop()
		coord.join(threads)
	print("done")

# Testing the degradation model: eval+testing
def run_adversarial_testing(cfg):
	'''
	Run testing of the trained model (direct test without any retraining, different from the two-fold-evaluation proposed in the paper)
	It will give the utility task accuracy and the privacy budget task accuracy
	'''
	# initialize multiplier_lst, logits_budget_lst_dct, loss_budget_lst_dct, which are used in both the graph and the session:
	# The depth multiplier list for creating different budget models ensemble (MobileNet with different depth.)
	multiplier_lst = [0.60 - i * 0.02 for i in range(FLAGS.NBudget)]
	# The dict of logits and loss for each different budget model to get accuracy
	logits_budget_lst_dct = {str(multiplier): [] for multiplier in multiplier_lst}
	loss_budget_lst_dct = {str(multiplier): [] for multiplier in multiplier_lst}
	# end initializing multiplier_lst, logits_budget_lst_dct, loss_budget_lst_dct.

	graph = tf.Graph()
	with graph.as_default():
		# placeholder inputs:
		videos_placeholder, utility_labels_placeholder, budget_labels_placeholder, dropout_placeholder, _ = placeholder_inputs(cfg['TEST']['BATCH_SIZE'] * FLAGS.GPU_NUM, cfg)

		# Compute Acc
		logits_utility_lst, logits_budget_lst = [], []

		with tf.variable_scope(tf.get_variable_scope()) as scope:
			# get the logits_budget and logits_utility on each gpu:
			for gpu_index in range(0, FLAGS.GPU_NUM):
				with tf.device('/gpu:%d' % gpu_index):
					print('/gpu:%d' % gpu_index)
					with tf.name_scope('%s_%d' % ('gpu', gpu_index)) as scope:
						videos = videos_placeholder[gpu_index * cfg['TEST']['BATCH_SIZE']:(gpu_index + 1) * cfg['TEST']['BATCH_SIZE']]
						utility_labels = utility_labels_placeholder[gpu_index * cfg['TEST']['BATCH_SIZE']:(gpu_index + 1) * cfg['TEST']['BATCH_SIZE']]
						budget_labels = budget_labels_placeholder[gpu_index * cfg['TEST']['BATCH_SIZE']:(gpu_index + 1) * cfg['TEST']['BATCH_SIZE']]
						_, _, _, logits_budget, logits_utility, _ = create_architecture_adversarial(cfg, cfg['TEST']['BATCH_SIZE'], multiplier_lst, logits_budget_lst_dct, loss_budget_lst_dct, scope, videos, utility_labels, budget_labels, dropout_placeholder)
						logits_budget_lst.append(logits_budget)
						logits_utility_lst.append(logits_utility)
						# print('len(logits_utility_lst):', len(logits_utility_lst))
						tf.get_variable_scope().reuse_variables()

		# concatnate the logits of each gpu:
		logits_utility = tf.concat(logits_utility_lst, 0)
		logits_budget = tf.concat(logits_budget_lst, 0)
		
		# count how many testing samples are classified correctly:
		right_count_utility_op = correct_num(logits_utility, utility_labels_placeholder)
		right_count_budget_op = correct_num(logits_budget, budget_labels_placeholder)

		# operations on each budget model:
		right_count_budget_op_lst = []
		for multiplier in multiplier_lst:
			# right count of each model:
			budget_logits_each_model = tf.concat(logits_budget_lst_dct['{}'.format(multiplier)], 0) # same budget model, concatenate over GPUs.
			right_count_op = correct_num(budget_logits_each_model, budget_labels_placeholder)
			right_count_budget_op_lst.append(right_count_op)

		videos_op, action_labels_op, actor_labels_op = create_videos_reading_ops(is_train=False, is_val=False, cfg=cfg)

	# session config:
	config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
	config.gpu_options.allow_growth = True

	# run session:
	with tf.Session(graph=graph, config=config) as sess:
		# initialization:
		init_op = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())
		sess.run(init_op)
		# initialization part should be put outside the multi-threads part! But why?
		
		# multi-threads:
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess=sess, coord=coord)
		
		# Create a saver for loading trained checkpoints:
		saver = tf.train.Saver(tf.trainable_variables())
		ckpt = tf.train.get_checkpoint_state(checkpoint_dir=ckpt_dir)
		# load trained checkpoints:
		if ckpt and ckpt.model_checkpoint_path:
			saver.restore(sess, ckpt.model_checkpoint_path)
			print('Session restored from trained model at {}!'.format(ckpt.model_checkpoint_path))
		else:
			raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), ckpt_dir)
		
		total_v = 0.0 # total number of testing samples

		test_correct_num_utility = 0.0 # how many testing samples get correct utility label prediction.
		test_correct_num_budget = 0.0 # how many testing samples get correct budget label prediction.
		test_correct_num_budget_lst = [0.0] * FLAGS.NBudget


		print('coord.should_stop():', coord.should_stop())
		try:
			c = 0
			batch_size = cfg['TEST']['BATCH_SIZE'] * FLAGS.GPU_NUM
			while not coord.should_stop():
				c += 1
				print('in while loop ', str(c))
				# input operations:
				test_videos, test_action_labels, test_actor_labels = sess.run([videos_op, action_labels_op, actor_labels_op])
				total_v += test_action_labels.shape[0]
				# padding:
				if test_videos.shape[0] < batch_size: # the last batch of testing data
					test_videos = np.pad(test_videos, ((0,batch_size-test_videos.shape[0]),(0,0),(0,0),(0,0),(0,0)), 'constant', constant_values=0)
					test_actor_labels = np.pad(test_actor_labels, ((0,batch_size-test_actor_labels.shape[0])), 'constant', constant_values=-1)
					test_action_labels = np.pad(test_action_labels, ((0,batch_size-test_action_labels.shape[0])), 'constant', constant_values=-1)
				# the padded videos will never be true, since it can never be classified as -1
				print('test_videos:', test_videos.shape)
				print('test_action_labels:', test_action_labels.shape)
				print('test_actor_labels:', test_actor_labels.shape)
				# placeholders:
				feed_dict = {videos_placeholder: test_videos, budget_labels_placeholder: test_actor_labels,
						utility_labels_placeholder: test_action_labels,
						dropout_placeholder: 1.0}
				# feed dorward:
				right_counts = sess.run([right_count_utility_op, right_count_budget_op] + right_count_budget_op_lst, feed_dict=feed_dict)
				print('right_counts:', right_counts)

				test_correct_num_utility += right_counts[0]
				test_correct_num_budget += right_counts[1]
				# testing acc for each one of N budget models:
				for i in range(FLAGS.NBudget):
					test_correct_num_budget_lst[i] += right_counts[i + 2]
				# end testing acc for each one of N budget models.
			# end try
		except tf.errors.OutOfRangeError:
			print('Done testing on all the examples')
		finally:
			coord.request_stop()

		# print and write file:
		test_result_str = ('test_acc_utility: {}, test_correct_num_utility: {}, total_v: {}\n'   
						'test_acc_budget: {}, test_correct_num_budget: {}, total_v: {}\n').format(
						test_correct_num_utility/total_v, test_correct_num_utility, total_v,
						test_correct_num_budget/total_v, test_correct_num_budget, total_v)
		print(test_result_str)
		if not os.path.exists(test_result_dir):
			os.makedirs(test_result_dir)
		test_result_file = open(test_result_dir+'/EvaluationResuls.txt', 'w')
		# write testing result to file:
		test_result_file.write(test_result_str)

		for i in range(FLAGS.NBudget):
			test_result_file.write('Budget{} test acc: {},\ttest_correct_num: {}\t: total_v: {}\n'.format(
				multiplier_lst[i], test_correct_num_budget_lst[i] / total_v,
				test_correct_num_budget_lst[i], total_v))
		# finish writing testing result to file.
		coord.join(threads)
		sess.close()

def main():
	# config:
	cfg = yaml.load(open('params_GRL.yml'))
	pp = pprint.PrettyPrinter()
	# pp.pprint(FLAGS.__flags)
	# pp.pprint(cfg)

	# adverserial training:
	run_adversarial_training(cfg)

	# testing learned fd:
	# run_adversarial_testing(cfg)

if __name__ == '__main__':
	main()