'''
kbeam
We should see validation and testing budget acc very low if we use testing mode BN in fb. If we use training mode BN in fb, the budget acc should be high.
'''
import sys, os, time, datetime, errno, pprint, itertools, yaml
sys.path.insert(0, '..')

import input_data
from modules.degradlNet import residualNet
from modules.budgetNet import budgetNet_kbeam
from modules.utilityNet import utilityNet
from loss import *
from utils import *
from validation import run_validation

from advtraining_flags import FLAGS
from common_flags import COMMON_FLAGS

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ["CUDA_VISIBLE_DEVICES"]=FLAGS.GPU_id
print('Using GPU:', FLAGS.GPU_id)
print('GPU_NUM:', FLAGS.GPU_NUM)

# config:
cfg = yaml.load(open('params_kbeam.yml'))

# job name:
_K = FLAGS._K
adversarial_job_name = '%s-%s-K%d' % ('kbeam', 'Rest' if FLAGS.use_budget_restarting else 'NoRest', _K)
dir_name = os.path.join(COMMON_FLAGS.hdd_dir, 'adversarial_training', adversarial_job_name)
# dir_name += datetime.datetime.now().strftime("-%Y%m%d_%H%M%S")
if not os.path.isdir(dir_name):
	os.mkdir(dir_name)
# dir name:
summary_dir = os.path.join(dir_name , 'summaries')
ckpt_dir = os.path.join(dir_name, 'ckpt_dir')
test_result_dir = os.path.join(dir_name, 'testing_results')
# end job name

def create_architecture_adversarial(batch_size, scope, videos, utility_labels, budget_labels, istraining_placeholder):
	'''
	Create the network architecture of the adversarial model in the graph
	Args:
		batch_size: int. Number of samples on a single GPU.
		videos: Input of the whole network. Original videos (center cropped).
		utility_labels: LT labels.
		budget_labels: Lb labels.
		istraining_placeholder: should be fed with True when training budgetNet.
	'''
	# fd part:
	degrad_videos = residualNet(videos, is_video=True)
	degrad_videos = avg_replicate(degrad_videos) if FLAGS.use_avg_replicate else degrad_videos
	# fd part ends
	# fT part:
	logits_utility = utilityNet(degrad_videos, wd=0.001)
	loss_utility = tower_loss_xentropy_sparse(logits_utility, utility_labels, use_weight_decay=True, name_scope=scope)
	# fT part ends
	# fb part:
	loss_budget, logits_budget = [0.0]*_K, [tf.zeros([batch_size, COMMON_FLAGS.NUM_CLASSES_BUDGET])]*_K
	for i in range(_K): # each element in loss_budget and logits_budget have the same graph structure.
		with tf.name_scope('%d' % _K):
			logits = budgetNet_kbeam(degrad_videos, K_id=i, depth_multiplier=0.6, is_training=istraining_placeholder)
			loss = tower_loss_xentropy_sparse(logits, budget_labels, use_weight_decay=False)
			logits_budget[i] += logits
			loss_budget[i] += loss
	# fd part ends.
	
	return loss_budget, loss_utility, logits_budget, logits_utility

def build_graph(batch_size):
	'''
	build the graph for adv training. The graph must be built with a fixed batch-size to support multi-gpu.
	Args:
		batch_size: int. Number of samples on a single GPU.
	'''
	# define graph
	graph = tf.Graph()
	with graph.as_default():
		# global step:
		global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
		# placeholder inputs:
		videos_placeholder, utility_labels_placeholder, budget_labels_placeholder, _, istraining_placeholder = \
										placeholder_inputs(batch_size * FLAGS.GPU_NUM)

		# initialize some lists, each element coresponds to one gpu:
		tower_grads_degrad = [[] for i in range(_K)] # degrade
		tower_grads_utility, logits_utility_lst,  loss_utility_lst= [], [], [] # utility
		tower_grads_budget, logits_budget_lst, loss_budget_lst = [[] for i in range(_K)], [[] for i in range(_K)], [[] for i in range(_K)] # budget

		# Optimizer for the 3 components respectively
		opt_degrad = tf.train.AdamOptimizer(FLAGS.degradation_lr)
		opt_utility = tf.train.AdamOptimizer(FLAGS.utility_lr)
		opt_budget = tf.train.AdamOptimizer(FLAGS.budget_lr)

		with tf.variable_scope(tf.get_variable_scope()):
			for gpu_index in range(0, FLAGS.GPU_NUM):
				with tf.device('/gpu:%d' % gpu_index):
					print('/gpu:%d' % gpu_index)
					with tf.name_scope('%s_%d' % ('gpu', gpu_index)) as scope:
						# placeholder inputs:
						videos = videos_placeholder[gpu_index * batch_size:(gpu_index + 1) * batch_size]
						utility_labels = utility_labels_placeholder[gpu_index * batch_size:(gpu_index + 1) * batch_size]
						budget_labels = budget_labels_placeholder[gpu_index * batch_size:(gpu_index + 1) * batch_size]
						# output of the graph:
						loss_budget, loss_utility, logits_budget, logits_utility = \
								create_architecture_adversarial(batch_size, scope, videos, utility_labels, budget_labels, istraining_placeholder)
						# Reuse variables for the next tower.
						tf.get_variable_scope().reuse_variables()
				
				### Append elements on each GPU to lists:
				## L_b:
				# budget:
				varlist_budget = [[] for i in range(_K)]
				for i in range(_K):
					# loss and logits:
					loss_budget_lst[i].append(loss_budget[i])
					logits_budget_lst[i].append(logits_budget[i])
					# gradients:
					varlist_budget[i] = [v for v in tf.trainable_variables() if "BudgetModule_{}".format(i) in v.name]
					grads_budget = opt_budget.compute_gradients(loss_budget[i], varlist_budget[i])
					tower_grads_budget[i].append(grads_budget)
				# degrade:
				# gradients:
				varlist_degrad = [v for v in tf.trainable_variables() if "DegradationModule" in v.name]
				for i in range(_K):
					grads_degrad = opt_degrad.compute_gradients(-loss_budget[i], varlist_degrad)
					tower_grads_degrad[i].append(grads_degrad)
				## LT:
				# loss and logits:
				loss_utility_lst.append(loss_utility)
				logits_utility_lst.append(logits_utility)
				# gradients:
				varlist_utility = [v for v in tf.trainable_variables() if any(x in v.name for x in ["UtilityModule"])]
				grads_utility = opt_utility.compute_gradients(loss_utility, varlist_utility + varlist_degrad)		
				tower_grads_utility.append(grads_utility)
				### End appending elements on each GPU to lists.
		
		### Average or concat Operations/Tnesors in a list to a single Operation/Tensor:
		## L_b
		# budget:
		loss_budget_op, accuracy_budget_op, right_count_budget_op = [None,]*_K, [None,]*_K, [None,]*_K
		zero_ops_budget, accum_ops_budget, apply_gradient_op_budget = [None,]*_K, [None,]*_K, [None,]*_K
		for i in range(_K):
			loss_budget_op[i] = tf.reduce_mean(loss_budget_lst[i], name='softmax') # Lb
			_logits_budget = tf.concat(logits_budget_lst[i], 0)
			accuracy_budget_op[i] = accuracy(_logits_budget, budget_labels_placeholder)
			right_count_budget_op[i] = correct_num(_logits_budget, budget_labels_placeholder)
			zero_ops_budget[i], accum_ops_budget[i], apply_gradient_op_budget[i] = create_grad_accum_for_late_update(
				opt_budget, tower_grads_budget[i], varlist_budget[i], FLAGS.n_minibatches, global_step, decay_with_global_step=False)
		# degrade:
		zero_ops_degrad, accum_ops_degrad, apply_gradient_op_degrad = [None,]*_K, [None,]*_K, [None,]*_K
		for i in range(_K):
			zero_ops_degrad[i], accum_ops_degrad[i], apply_gradient_op_degrad[i] = create_grad_accum_for_late_update(
				opt_degrad, tower_grads_degrad[i], varlist_degrad, FLAGS.n_minibatches, global_step, decay_with_global_step=True)

		## L_T
		loss_utility_op = tf.reduce_mean(loss_utility_lst, name='softmax') # mean loss over all GPU
		logits_utility = tf.concat(logits_utility_lst, 0) # Concatenate the logits over all GPU
		accuracy_utility_op = accuracy(logits_utility, utility_labels_placeholder) # acc
		right_count_utility_op = correct_num(logits_utility, utility_labels_placeholder) # right count
		zero_ops_utility, accum_ops_utility, apply_gradient_op_utility = create_grad_accum_for_late_update(
			opt_utility, tower_grads_utility, varlist_utility+varlist_degrad, FLAGS.n_minibatches, global_step, decay_with_global_step=False)
		### End averaging or concatenating Operations/Tnesors in a list to a single Operation/Tensor.
		
		# operations for placeholder inputs:
		tr_videos_op, tr_action_labels_op, tr_actor_labels_op = create_videos_reading_ops(is_train=True, is_val=False, GPU_NUM=FLAGS.GPU_NUM, BATCH_SIZE=batch_size)
		test_videos_op, test_action_labels_op, test_actor_labels_op = create_videos_reading_ops(is_train=False, is_val=False, GPU_NUM=FLAGS.GPU_NUM, BATCH_SIZE=batch_size)

		# initialize:
		init_op = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())

		# bn varlist
		varlist_bn = [g for g in tf.global_variables() if 'moving_mean' in g.name]
		varlist_bn += [g for g in tf.global_variables() if 'moving_variance' in g.name]
		# print(varlist_bn)

	return (graph, init_op, 
			zero_ops_degrad, accum_ops_degrad, apply_gradient_op_degrad, 
			zero_ops_budget, accum_ops_budget, apply_gradient_op_budget, 
			zero_ops_utility, accum_ops_utility, apply_gradient_op_utility, 
			loss_budget_op, accuracy_budget_op, right_count_budget_op, 
			loss_utility_op, accuracy_utility_op, right_count_utility_op,
			tr_videos_op, tr_action_labels_op, tr_actor_labels_op, 
			videos_placeholder, utility_labels_placeholder, budget_labels_placeholder, istraining_placeholder, 
			varlist_budget, varlist_utility, varlist_degrad, varlist_bn)

# Training set for traning, validation set for validation.
def run_adversarial_training(start_from_trained_model):
	'''
	Algorithm 1 in the paper
	'''
	# Save ckpt of adv-training process in this directory:
	if not os.path.exists(ckpt_dir):
		os.makedirs(ckpt_dir)
	# Save summary files in this dir:
	if not os.path.exists(summary_dir):
		os.makedirs(summary_dir)
	train_summary_file = open(summary_dir + '/train_summary.txt', 'a', buffering=1)
	validation_summary_file = open(summary_dir + '/validation_summary.txt', 'a', buffering=1)
	model_restarting_summary_file = open(summary_dir + '/model_restarting_summary.txt', 'a', buffering=1)

	(graph, init_op, 
		zero_ops_degrad, accum_ops_degrad, apply_gradient_op_degrad, 
		zero_ops_budget, accum_ops_budget, apply_gradient_op_budget, 
		zero_ops_utility, accum_ops_utility, apply_gradient_op_utility, 
		loss_budget_op, accuracy_budget_op, right_count_budget_op, 
		loss_utility_op, accuracy_utility_op, right_count_utility_op,
		tr_videos_op, tr_action_labels_op, tr_actor_labels_op, 
		videos_placeholder, utility_labels_placeholder, budget_labels_placeholder, istraining_placeholder, 
		varlist_budget, varlist_utility, varlist_degrad, varlist_bn) = build_graph(cfg['TRAIN']['BATCH_SIZE'])

	
	# session config:
	config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
	# config.gpu_options.allow_growth = True # Don't do this!
	# run session:
	with tf.Session(graph=graph, config=config) as sess:
		# saver for saving models:
		saver = tf.train.Saver(var_list=tf.trainable_variables()+varlist_bn, max_to_keep=5)
		
		sess.run(init_op)

		# multi-threads:
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess=sess, coord=coord)
		
		# load ckpts: 
		if not start_from_trained_model: # load ckpts from pretrained fd and fT.(By run_pretraining_degrad and run_pretraining_utility functions.)
			# fd part:
			restore_model_ckpt(sess, os.path.join(COMMON_FLAGS.pretrain_dir, 'pretrained_fd'), varlist_degrad)
			# fT part:
			restore_model_ckpt(sess, os.path.join(COMMON_FLAGS.pretrain_dir, 'pretrained_kbeam/fT'), varlist_utility)
			# fb part:
			for i in range(_K):
				restore_model_ckpt(sess, os.path.join(COMMON_FLAGS.pretrain_dir, 'pretrained_kbeam/fb_k%d' % i), varlist_budget[i])
			
		else: # load ckpts from previous training stage of this run_adversarial_training function.
			restore_model_ckpt(sess, ckpt_dir, tf.trainable_variables())

		# Adversarial training loop:
		_idx_min = 0 # initialize _idx_min randomly
		for step in range(cfg['TRAIN']['TOP_MAXSTEP']):
			
			# Part 3: train Fb using L_b (cross entropy)

			# max step: optimize theta_d using L_b(X,Y_B)
			for L_b_max_step in range(0, cfg['TRAIN']['L_B_MAX_PART_STEP']):
				start_time = time.time()
				acc_util_lst, acc_budget_lst, loss_utility_lst, loss_budget_lst = [], [], [], []
				sess.run(zero_ops_degrad)
				# accumulating gradient for late update:
				for _ in itertools.repeat(None, FLAGS.n_minibatches):
					# placeholder inputs:
					tr_videos, tr_action_labels, tr_actor_labels = sess.run([tr_videos_op, tr_action_labels_op, tr_actor_labels_op])
					# run operations:
					_, acc_util_value, acc_budget_value, loss_utility_value, loss_budget_value = sess.run(
									[accum_ops_degrad[_idx_min], accuracy_utility_op, accuracy_budget_op[_idx_min], loss_utility_op, loss_budget_op[_idx_min]],
									feed_dict={videos_placeholder: tr_videos,
												utility_labels_placeholder: tr_action_labels,
												budget_labels_placeholder: tr_actor_labels,
												istraining_placeholder: True})
					# append loss and acc for budget model:
					acc_util_lst.append(acc_util_value)
					acc_budget_lst.append(acc_budget_value)
					loss_utility_lst.append(loss_utility_value)
					loss_budget_lst.append(loss_budget_value)
				# finish accumulating gradient for late update
				# after accumulating gradient, do the update on fd:
				_ = sess.run([apply_gradient_op_degrad[_idx_min]]) # update only one wd
				# finish update on fd

				assert not np.isnan(np.mean(loss_budget_value)), 'Model diverged with loss = NaN'

				# loss summary:
				if L_b_max_step % cfg['TRAIN']['L_B_MAX_PRINT_STEP'] == 0:
					loss_summary = 'Alternating Training (Budget L_b MAX), Step: {:2d}, L_b_max_step: {:2d} time: {:.2f}, ' \
								'training utility accuracy: {:.5f}, training budget accuracy: {:.5f}, ' \
								'utility loss: {:.8f}, budget loss: {:.8f}'.format(
								step, L_b_max_step, time.time() - start_time, 
								np.mean(acc_util_lst), np.mean(acc_budget_lst),
								np.mean(loss_utility_lst), np.mean(loss_budget_lst)
								)

					print(loss_summary)
					train_summary_file.write(loss_summary + '\n')
				# end loss summary
			print()
			# End max step

			# min step: optimize theta_b using L_b(X,Y_B)
			for L_b_min_step in range(0, cfg['TRAIN']['L_B_MIN_PART_STEP']):
				start_time = time.time()
				sess.run(zero_ops_budget)
				acc_budget_lst, loss_budget_lst = [[] for i in range(_K)], [[] for i in range(_K)]
				acc_budget_lst_mean, loss_budget_lst_mean = [None]*_K, [None]*_K
				# accumulating gradient for late update:
				for _ in itertools.repeat(None, FLAGS.n_minibatches):
					# placeholder inputs:
					tr_videos, tr_actor_labels = sess.run([tr_videos_op, tr_actor_labels_op])
					# run operations:
					temp_sess_run_return_list = sess.run(accum_ops_budget + accuracy_budget_op + loss_budget_op,
								feed_dict={videos_placeholder: tr_videos,
											budget_labels_placeholder: tr_actor_labels,
											istraining_placeholder: True})
					acc_budget_value = temp_sess_run_return_list[_K : 2*_K]
					loss_budget_value = temp_sess_run_return_list[2*_K : 3*_K]
					# append loss and acc for budget model:
					for i in range(_K):
						acc_budget_lst[i].append(acc_budget_value[i])
						loss_budget_lst[i].append(loss_budget_value[i])
				# finish accumulating gradient for late update
				# find acc and loss mean across all gpus:
				for i in range(_K):
					acc_budget_lst_mean[i] = np.mean(acc_budget_lst[i])
					loss_budget_lst_mean[i] = np.mean(loss_budget_lst[i])
				# find min loss:
				_idx_min = np.argmin(loss_budget_lst_mean)
				
				assert not np.isnan(loss_budget_lst_mean[_idx_min]), 'Model diverged with loss = NaN'

				# Monitoring fb using training set
				if L_b_min_step % cfg['TRAIN']['MONITOR_STEP'] == 0:
					if acc_budget_lst_mean[_idx_min] >= FLAGS.highest_budget_acc_val:
						print('pass budget acc bar!\n')
						train_summary_file.write('pass budget acc bar!\n')
						break
				# End monitoring fb on training set.

				# after accumulating gradient, do the update on fb, if it didn't pass the budget acc bar:
				sess.run([apply_gradient_op_budget]) # update all k wb's
				# finish update on fb

				# loss summary:
				if L_b_min_step % cfg['TRAIN']['MONITOR_STEP'] == 0:
					loss_summary = 'Alternating Training (Budget L_b MIN), ' \
								'Step: %2d, L_b_min_step: %4d, time: %.4f, ' \
								'training budget accuracy: %s, budget loss: %s, ' \
								'min_idx: %1d' % (
								step, L_b_min_step, time.time() - start_time, 
								acc_budget_lst_mean, loss_budget_lst_mean,
								_idx_min)

					print(loss_summary)
					train_summary_file.write(loss_summary + '\n')
				# end loss summary	
			print('')
			# End min step

			train_summary_file.write('\n')
			# End part 3
			
			# Part 2: End-to-end train Ft and Fd using L_T
			# for L_T_step in range(0, cfg['TRAIN']['L_T_MAXSTEP']):
			L_T_step = 0
			plateau_counter = 0
			test_acc_util_best = -1
			while(plateau_counter<cfg['TRAIN']['L_T_PLATEAUSTEP'] and L_T_step<cfg['TRAIN']['L_T_MAXSTEP']):

				# Monitoring LT using validation set:
				if L_T_step % cfg['TRAIN']['MONITOR_STEP'] == 0:
					print('L_T_step %d monitoring target task:' % L_T_step)
					train_summary_file.write('L_T_step %d monitoring target task: \n' % L_T_step)
					test_correct_num_lst, test_acc_lst, total_v = run_validation(sess=sess, 
						right_count_op_list=[right_count_utility_op] + right_count_budget_op, 
						placeholder_list=[videos_placeholder, utility_labels_placeholder, budget_labels_placeholder, istraining_placeholder], 
						batch_size = cfg['TRAIN']['BATCH_SIZE']*FLAGS.GPU_NUM, 
						dataset = 'val', 
						istraining=True)
					# plateau
					test_acc_util = test_acc_lst[0]
					if test_acc_util <= test_acc_util_best:
						plateau_counter += 1
					else:
						plateau_counter = 0
						test_acc_util_best = test_acc_util
					# print and write summary:
					test_summary = ('val_correct_num_utility: %s, val_correct_num_budget: %s, total_v: %d\n'   
									'plateau_counter: %d, test_acc_util_best: %.2f\n'
									% (test_correct_num_lst[0], test_correct_num_lst[1:], total_v, plateau_counter, test_acc_util_best))
					print(test_summary)
					train_summary_file.write(test_summary + '\n')
					# breaking condition: (if performance on L_T is still good)
					
					if test_acc_util >= FLAGS.highest_util_acc_val:
						print('pass utility acc bar!\n')
						train_summary_file.write('pass utility acc bar!\n')
						break
					
					
				# End of monitoring LT

				# Optimizing LT (if necessary) using training set: (This is one batch=FLAGS.n_minibatches, each minibatch has FLAGS.GPU_NUM*cfg['TRAIN']['BATCH_SIZE'] video clips.)               
				start_time = time.time()
				sess.run(zero_ops_utility)
				acc_util_lst, acc_budget_lst, loss_utility_lst, loss_budget_lst = [], [], [], []
				# accumulating gradient for late update:
				for _ in itertools.repeat(None, FLAGS.n_minibatches):
					tr_videos, tr_action_labels, tr_actor_labels = sess.run([tr_videos_op, tr_action_labels_op, tr_actor_labels_op])
					_, acc_util_value, acc_budget_value, loss_utility_value, loss_budget_value = sess.run(
								[accum_ops_utility, accuracy_utility_op, accuracy_budget_op[_idx_min], loss_utility_op, loss_budget_op[_idx_min]],
								feed_dict={videos_placeholder: tr_videos,
											utility_labels_placeholder: tr_action_labels,
											budget_labels_placeholder: tr_actor_labels,
											istraining_placeholder: True}, 
								options = tf.RunOptions(report_tensor_allocations_upon_oom = True))
					acc_util_lst.append(acc_util_value)
					acc_budget_lst.append(acc_budget_value)
					loss_utility_lst.append(loss_utility_value)
					loss_budget_lst.append(loss_budget_value)
				# finish accumulating gradient for late update
				# after accumulating gradient, do the update on fT and fd:
				sess.run([apply_gradient_op_utility])
				# finish update on fT and fd

				assert not np.isnan(np.mean(loss_utility_lst)), 'Model diverged with loss = NaN'

				# loss summary:
				loss_summary = 'min LT (Utility), Step: {:4d}, L_T_step: {:4d}, time: {:.2f}, ' \
							'training utility accuracy: {:.5f}, training budget accuracy: {:.5f}, ' \
							'utility loss: {:.8f}, budget loss: {:.8f}'.format(
							step, L_T_step, time.time() - start_time,
							np.mean(acc_util_lst), np.mean(acc_budget_lst),
							np.mean(loss_utility_lst), np.mean(loss_budget_lst)
							)

				print('\n' + loss_summary + '\n')
				train_summary_file.write(loss_summary + '\n')
				# end of loss summary

				
				L_T_step += 1
				# End of optimizing LT.
			
			print('')
			train_summary_file.write('\n')
			# End part 2

			# Do validation (on validation set):
			if step % cfg['TRAIN']['VAL_STEP'] == 0:
				print('step %d validation: \n' % step)
				validation_summary_file.write('step %d validation: \n' % step)
				test_correct_num_lst, test_acc_lst, total_v = run_validation(sess=sess, 
					right_count_op_list=[right_count_utility_op] + right_count_budget_op, 
					placeholder_list=[videos_placeholder, utility_labels_placeholder, budget_labels_placeholder, istraining_placeholder], 
					batch_size = cfg['TRAIN']['BATCH_SIZE']*FLAGS.GPU_NUM, 
					dataset = 'val', 
					istraining=True)
				# print and write summary:
				test_summary = ('val_correct_num_utility: %s, total_v: %d\n'   
								'val_correct_num_budget: %s, total_v: %d\n'
								% (test_correct_num_lst[0],total_v,test_correct_num_lst[1:],total_v))
				print(test_summary)
				validation_summary_file.write(test_summary + '\n')

			# End evaluation
			# Save ckpt for kb_adversarial learning:
			if step % cfg['TRAIN']['SAVE_STEP'] == 0 or (step + 1) == cfg['TRAIN']['TOP_MAXSTEP']:
				checkpoint_path = os.path.join(ckpt_dir, 'model.ckpt')
				print('+++++++++++++ saving model to %s +++++++++++++' % checkpoint_path)
				saver.save(sess, checkpoint_path, global_step=step)
			# End evaluation

		train_summary_file.close()
		validation_summary_file.close()
		coord.request_stop()
		coord.join(threads)
	print("done")

# Testing the degradation model: eval+testing
def run_adversarial_testing():
	start_time = time.time()
	(graph, init_op, 
		zero_ops_degrad, accum_ops_degrad, apply_gradient_op_degrad, 
		zero_ops_budget, accum_ops_budget, apply_gradient_op_budget, 
		zero_ops_utility, accum_ops_utility, apply_gradient_op_utility, 
		loss_budget_op, accuracy_budget_op, right_count_budget_op, 
		loss_utility_op, accuracy_utility_op, right_count_utility_op,
		tr_videos_op, tr_action_labels_op, tr_actor_labels_op, 
		videos_placeholder, utility_labels_placeholder, budget_labels_placeholder, istraining_placeholder, 
		varlist_budget, varlist_utility, varlist_degrad, varlist_bn) = build_graph(cfg['TEST']['BATCH_SIZE'])
		
	if not os.path.exists(test_result_dir):
		os.makedirs(test_result_dir)
	test_result_file = open(test_result_dir+'/EvaluationResuls.txt', 'w', buffering=1)

	# session config:
	config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
	# config.gpu_options.allow_growth = True

	# run session:
	with tf.Session(graph=graph, config=config) as sess:
		# initialization:
		init_op = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())
		sess.run(init_op)
		# initialization part should be put outside the multi-threads part! But why?
				
		# loading trained checkpoints:
		restore_model_ckpt(sess, ckpt_dir, tf.trainable_variables()+varlist_bn)

		test_correct_num_lst, test_acc_lst, total_v = run_validation(sess=sess, right_count_op_list=[right_count_utility_op] + right_count_budget_op, 
			placeholder_list=[videos_placeholder, utility_labels_placeholder, budget_labels_placeholder, istraining_placeholder], 
			batch_size=cfg['TEST']['BATCH_SIZE']*FLAGS.GPU_NUM, 
			dataset='test', 
			istraining=True)
		# print and write summary:
		test_summary = ('test_acc_utility: %s, test_correct_num_utility: %s, total_v: %d\n'   
						'test_acc_budget: %s, test_correct_num_budget: %s, total_v: %d\n'
						% (test_acc_lst[0], test_correct_num_lst[0], total_v, 
						test_acc_lst[1:], test_correct_num_lst[1:], total_v))
		print(test_summary)
		test_result_file.write(test_summary + '\n')

		sess.close()

	finish_time = time.time()
	print(finish_time-start_time)
	test_result_file.write(str(finish_time-start_time))

def main():
	# adverserial training:
	start_from_trained_model = False # if False, load ckpts from pretrained fd, fb and fT; 
	# start_from_trained_model = True # if True, load ckpts from previous training stage of this run_adversarial_training function.
	run_adversarial_training(start_from_trained_model)

	# testing learned fd:
	run_adversarial_testing()

if __name__ == '__main__':
	main()