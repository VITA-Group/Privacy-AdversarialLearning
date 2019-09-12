'''
pretrain_fb_kbeam
'''
import sys, time, os, itertools
sys.path.insert(0, '..')
 
import input_data 
from modules.degradlNet import residualNet
from modules.budgetNet import budgetNet_kbeam
from loss import *
from utils import *
from validation import run_validation

from common_flags import COMMON_FLAGS
from pretrain_flags import FLAGS

os.environ["CUDA_VISIBLE_DEVICES"]=FLAGS.GPU_id
print('Using GPU:', FLAGS.GPU_id)
print('GPU_NUM:', FLAGS.GPU_NUM)

_K = 8

MAX_STEPS = 150
SAVE_STEP = 50
VAL_STEP = 10
PRINT_STEP = 10
TRAIN_BATCH_SIZE = 2

def create_architecture_adversarial(scope, videos, budget_labels, istraining_placeholder):
	'''
	Create the architecture of the adversarial model in the graph
	is_training: whether it is in the adversarial training part. (include testing, not two-fold)
	'''
	# fd part:
	degrad_videos = residualNet(videos, is_video=True)
	degrad_videos = avg_replicate(degrad_videos) if FLAGS.use_avg_replicate else degrad_videos
	# fd part ends

	# fb part:
	loss_budget, logits_budget = [0.0]*_K, [tf.zeros([TRAIN_BATCH_SIZE, COMMON_FLAGS.NUM_CLASSES_BUDGET])]*_K
	for i in range(_K): # each element in loss_budget and logits_budget have the same graph structure.
		with tf.name_scope('%d' % _K) as scope:
			logits = budgetNet_kbeam(degrad_videos, K_id=FLAGS.base_idx+i, is_training=istraining_placeholder, depth_multiplier=0.6)
			loss = tower_loss_xentropy_sparse(logits, budget_labels, use_weight_decay=False)
			logits_budget[i] += logits
			loss_budget[i] += loss
	# fd part ends.
	
	return loss_budget, logits_budget

# Training set for traning, validation set for validation.
# lambda: weight for L1 loss (degrade model L1 loss)
def pretrain_fb_kbeam(start_from_trained_model):
	'''
	pretrain_fb_kbeam
	Args:
        start_from_trained_model: boolean. If False, use random initialized fb. If true, use pretrained fb.
	'''
	# mkdir:
	degradation_ckpt_dir = os.path.join(COMMON_FLAGS.pretrain_dir, 'degradation_models')
	budget_ckpt_dir = ['']*_K
	for i in range(_K):
		budget_ckpt_dir[i] = os.path.join(COMMON_FLAGS.pretrain_dir, 'budget_k%d_new' % (FLAGS.base_idx+i))
		if not os.path.isdir(budget_ckpt_dir[i]):
			os.mkdir(budget_ckpt_dir[i])

	# define graph
	graph = tf.Graph()
	with graph.as_default():
		# global step:
		global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
		# placeholder inputs:
		videos_placeholder, utility_labels_placeholder, budget_labels_placeholder, dropout_placeholder, istraining_placeholder = \
			placeholder_inputs(TRAIN_BATCH_SIZE * FLAGS.GPU_NUM)

		# initialize some lists, each element coresponds to one gpu:
		tower_grads_budget, logits_budget_lst, loss_budget_lst = [[] for i in range(_K)], [[] for i in range(_K)], [[] for i in range(_K)] # budget

		# Optimizer for the 3 components respectively
		opt_budget = tf.train.AdamOptimizer(FLAGS.budget_lr)

		with tf.variable_scope(tf.get_variable_scope()):
			for gpu_index in range(0, FLAGS.GPU_NUM):
				with tf.device('/gpu:%d' % gpu_index):
					print('/gpu:%d' % gpu_index)
					with tf.name_scope('%s_%d' % ('gpu', gpu_index)) as scope:
						# placeholder inputs:
						videos = videos_placeholder[gpu_index * TRAIN_BATCH_SIZE:(gpu_index + 1) * TRAIN_BATCH_SIZE]
						budget_labels = budget_labels_placeholder[gpu_index * TRAIN_BATCH_SIZE:(gpu_index + 1) * TRAIN_BATCH_SIZE]
						# output of the graph:
						loss_budget, logits_budget = \
									create_architecture_adversarial(scope, videos, budget_labels, istraining_placeholder)
						# Reuse variables for the next tower.
						tf.get_variable_scope().reuse_variables()
				
				# degrade:
				varlist_degrad = [v for v in tf.trainable_variables() if "DegradationModule" in v.name]
				# bn varlist
				varlist_bn = [g for g in tf.global_variables() if 'moving_mean' in g.name]
				varlist_bn += [g for g in tf.global_variables() if 'moving_variance' in g.name]
				# budget:
				varlist_budget = [[] for i in range(_K)]
				varlist_budget_bn = [[] for i in range(_K)]
				### Append elements on each GPU to lists:
				for i in range(_K):
					# loss and logits:
					loss_budget_lst[i].append(loss_budget[i])
					logits_budget_lst[i].append(logits_budget[i])
					# gradients:
					varlist_budget[i] = [v for v in tf.trainable_variables() if "BudgetModule_%d" % (FLAGS.base_idx+i) in v.name]
					varlist_budget_bn[i] = [v for v in varlist_bn if "BudgetModule_%d" % (FLAGS.base_idx+i) in v.name]
					grads_budget = opt_budget.compute_gradients(loss_budget[i], varlist_budget[i])
					tower_grads_budget[i].append(grads_budget)
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
		### End averaging or concatenating Operations/Tnesors in a list to a single Operation/Tensor.
		
		# operations for placeholder inputs:
		tr_videos_op, _, tr_actor_labels_op = create_videos_reading_ops(is_train=True, is_val=False, GPU_NUM=FLAGS.GPU_NUM, BATCH_SIZE=TRAIN_BATCH_SIZE)
		# saver and summary files:
		saver_kb = [None]*_K
		for i in range(_K): # saver used for saving pretrained fb.
			saver_kb[i] = tf.train.Saver(var_list=varlist_budget[i]+varlist_budget_bn[i], max_to_keep=1)
		
		init_op = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())

	# session config:
	config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
	# run session:
	with tf.Session(graph=graph, config=config) as sess:
		# initialize:
		sess.run(init_op)
		# multi-threads:
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess=sess, coord=coord)
		# restore d net:
		restore_model_ckpt(sess=sess, ckpt_dir=degradation_ckpt_dir, varlist=varlist_degrad)
		# restore b net:
		if start_from_trained_model:
			for i in range(_K):
				restore_model_ckpt(sess, budget_ckpt_dir[i], varlist_budget[i]+varlist_budget_bn[i])
				
		# train
		for step in range(0, MAX_STEPS):
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
			
			sess.run([apply_gradient_op_budget]) # update all k wb's
			# finish update on fb

			# loss summary:
			if step % PRINT_STEP == 0:
				loss_summary = 'step: %4d, time: %.4f, ' \
							'training budget accuracy: %s, budget loss: %s' % (
							step, time.time() - start_time, 
							acc_budget_lst_mean, loss_budget_lst_mean)
				print(loss_summary)
			# end loss summary	

			if step % VAL_STEP == 0:
				test_correct_num_lst, test_acc_lst, total_v = run_validation(sess=sess, 
					right_count_op_list=right_count_budget_op,
					placeholder_list=[videos_placeholder, utility_labels_placeholder, budget_labels_placeholder, dropout_placeholder, istraining_placeholder],
					batch_size=TRAIN_BATCH_SIZE*FLAGS.GPU_NUM, dataset='val', istraining=True)
				test_summary = "Step: %d, validation budget correct num: %s, accuracy: %s" % (
                        step, test_correct_num_lst, test_acc_lst)
				print(test_summary)
				# bn_temp_value = sess.run(varlist_bn[-1])
				# print('bn_temp_value:', bn_temp_value.shape, bn_temp_value[0:5])

			# save model:
			if step % SAVE_STEP == 0 or (step + 1) == MAX_STEPS:
				for i in range(_K):
					saver_kb[i].save(sess, os.path.join(budget_ckpt_dir[i], 'pretrained_fb_k%d.ckpt' % i), global_step=step) 
		# End min step
		# End part 3

		coord.request_stop()
		coord.join(threads)
	print("done")


if __name__ == '__main__':
	start_from_trained_model = False
	pretrain_fb_kbeam(start_from_trained_model)