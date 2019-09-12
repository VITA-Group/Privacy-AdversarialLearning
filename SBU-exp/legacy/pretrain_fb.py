'''
pretrain_fb_kbeam
'''
import sys, time, itertools, os
sys.path.insert(0, '..')
 
import input_data 
from degradlNet import residualNet
from budgetNet import budgetNet
from loss import *
from utils import *
from validation import run_validation

from common_flags import COMMON_FLAGS
from pretrain_flags import FLAGS

os.environ["CUDA_VISIBLE_DEVICES"]=FLAGS.GPU_id
print('Using GPU:', FLAGS.GPU_id)
print('GPU_NUM:', FLAGS.GPU_NUM)


MAX_STEPS = 1000
SAVE_STEP = 50
PRINT_STEP = 1
VAL_STEP = 2
TRAIN_BATCH_SIZE = 2

use_pretrained_model = False # if True, load ckpts from pretrained fb; if False: random initialize fb.


def create_architecture_adversarial(scope, videos, budget_labels, istraining_placeholder):
	'''
	Create the architecture of the adversarial model in the graph
	Args:
		is_training: whether it is in the adversarial training part. (include testing, not two-fold)
	'''
	# fd part:
	degrad_videos = residualNet(videos, is_video=True)
	degrad_videos = _avg_replicate(degrad_videos) if FLAGS.use_avg_replicate else degrad_videos
	# fd part ends

	# fb part:
	loss_budget = 0.0
	logits_budget = tf.zeros([TRAIN_BATCH_SIZE, COMMON_FLAGS.NUM_CLASSES_BUDGET])
	logits = budgetNet(degrad_videos, depth_multiplier=0.6, is_training=istraining_placeholder)
	loss = tower_loss_xentropy_sparse(logits, budget_labels, use_weight_decay=False, name_scope=scope)
	logits_budget += logits
	loss_budget += loss
	# fd part ends.
	
	return loss_budget, logits_budget

# Training set for traning, validation set for validation.
# lambda: weight for L1 loss (degrade model L1 loss)
def pretrain_fb(start_from_trained_model):
	'''
	pretrain_fb_kbeam
	'''
	degradation_ckpt_dir = os.path.join(COMMON_FLAGS.pretrain_dir, 'degradation_models')
	budget_ckpt_dir = os.path.join(COMMON_FLAGS.pretrain_dir, 'budget_models' + '_temp')
	if not os.path.isdir(budget_ckpt_dir):
		os.mkdir(budget_ckpt_dir)

	# define graph
	graph = tf.Graph()
	with graph.as_default():
		# global step:
		global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
		# placeholder inputs:
		videos_placeholder, utility_labels_placeholder, budget_labels_placeholder, dropout_placeholder, istraining_placeholder = \
										placeholder_inputs(TRAIN_BATCH_SIZE * FLAGS.GPU_NUM)

		# initialize some lists, each element coresponds to one gpu:
		tower_grads_budget, logits_budget_lst, loss_budget_lst = [], [], [] # budget

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
				
				### Append elements on each GPU to lists:
				varlist_degrad = [v for v in tf.trainable_variables() if "DegradationModule" in v.name]
				# budget:
				varlist_budget = []
				# loss and logits:
				loss_budget_lst.append(loss_budget)
				logits_budget_lst.append(logits_budget)
				# varlist:
				# budget
				varlist_budget = [v for v in tf.trainable_variables() if "BudgetModule" in v.name]
				# bn varlist
				varlist_bn = [g for g in tf.global_variables() if 'moving_mean' in g.name]
				varlist_bn += [g for g in tf.global_variables() if 'moving_variance' in g.name]
				# gradients:
				grads_budget = opt_budget.compute_gradients(loss_budget, varlist_budget)
				tower_grads_budget.append(grads_budget)
				### End appending elements on each GPU to lists.
		
		### Average or concat Operations/Tnesors in a list to a single Operation/Tensor:
		## L_b
		# budget:
		loss_budget_op = tf.reduce_mean(loss_budget_lst, name='softmax') # Lb
		_logits_budget = tf.concat(logits_budget_lst, 0)
		accuracy_budget_op = accuracy(_logits_budget, budget_labels_placeholder)
		right_count_budget_op = correct_num(_logits_budget, budget_labels_placeholder)
		zero_ops_budget, accum_ops_budget, apply_gradient_op_budget = create_grad_accum_for_late_update(
			opt_budget, tower_grads_budget, varlist_budget, FLAGS.n_minibatches, global_step, decay_with_global_step=False)
		### End averaging or concatenating Operations/Tnesors in a list to a single Operation/Tensor.
		
		# operations for placeholder inputs:
		tr_videos_op, _, tr_actor_labels_op = create_videos_reading_ops(is_train=True, is_val=False, GPU_NUM=FLAGS.GPU_NUM, BATCH_SIZE=TRAIN_BATCH_SIZE)
		# saver and summary files:
		saver = tf.train.Saver(var_list=varlist_budget+varlist_bn, max_to_keep=1)

	# session config:
	config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
	config.gpu_options.allow_growth = True
	# run session:
	with tf.Session(graph=graph, config=config) as sess:
		# multi-threads:
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess=sess, coord=coord)
		# initialize:
		init_op = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())
		sess.run(init_op)
		# restore d net:
		restore_model_ckpt(sess, degradation_ckpt_dir, varlist_degrad)
		# restore b net:
		if start_from_trained_model:
			restore_model_ckpt(sess, budget_ckpt_dir, varlist_budget+varlist_bn)
		
		# train
		for step in range(0, MAX_STEPS):
			start_time = time.time()
			sess.run(zero_ops_budget)
			acc_budget_lst, loss_budget_lst = [], []
			# accumulating gradient for late update:
			for _ in itertools.repeat(None, FLAGS.n_minibatches):
				# placeholder inputs:
				tr_videos, tr_actor_labels = sess.run([tr_videos_op, tr_actor_labels_op])
				# run operations:
				_, acc_budget_value, loss_budget_value = sess.run([accum_ops_budget, accuracy_budget_op, loss_budget_op],
							feed_dict={videos_placeholder: tr_videos,
										budget_labels_placeholder: tr_actor_labels,
										istraining_placeholder: True})
				# append loss and acc for budget model:
				acc_budget_lst.append(acc_budget_value)
				loss_budget_lst.append(loss_budget_value)
			# finish accumulating gradient for late update
			# find acc and loss mean across all gpus:
			
			assert not np.isnan(np.mean(loss_budget_lst)), 'Model diverged with loss = NaN'

			sess.run([apply_gradient_op_budget]) # update all k wb's
			# finish update on fb

			# loss summary:
			if step % PRINT_STEP == 0:
				loss_summary = 'step: %4d, time: %.4f, ' \
							'training budget accuracy: %s, budget loss: %s, ' \
							% (step, time.time() - start_time, 
							np.mean(acc_budget_lst), np.mean(loss_budget_lst),)
				print(loss_summary)
			# end loss summary	

			# val:
			if step % VAL_STEP == 0:
				# val set:
				test_correct_num_lst, test_acc_lst, total_v = run_validation(sess=sess, 
					right_count_op_list=[right_count_budget_op],
					placeholder_list=[videos_placeholder, utility_labels_placeholder, budget_labels_placeholder, dropout_placeholder, istraining_placeholder],
					batch_size=TRAIN_BATCH_SIZE*FLAGS.GPU_NUM, dataset='val')
				test_summary = "Step: %d, validation budget correct num: %s, accuracy: %s" % (
                        step, test_correct_num_lst, test_acc_lst)
				print(test_summary)
				# test set:
				test_correct_num_lst, test_acc_lst, total_v = run_validation(sess=sess, 
					right_count_op_list=[right_count_budget_op],
					placeholder_list=[videos_placeholder, utility_labels_placeholder, budget_labels_placeholder, dropout_placeholder, istraining_placeholder],
					batch_size=TRAIN_BATCH_SIZE*FLAGS.GPU_NUM, dataset='test')
				test_summary = "Step: %d, testing budget correct num: %s, accuracy: %s" % (
                        step, test_correct_num_lst, test_acc_lst)
				print(test_summary)

			if step % SAVE_STEP == 0 or (step + 1) == MAX_STEPS: 
				saver.save(sess, os.path.join(budget_ckpt_dir, 'pretrained_fb.ckpt'), global_step=step) 
		# End min step
		# End part 3

		coord.request_stop()
		coord.join(threads)
	print("done")


if __name__ == '__main__':
	start_from_trained_model = False
	pretrain_fb(start_from_trained_model)