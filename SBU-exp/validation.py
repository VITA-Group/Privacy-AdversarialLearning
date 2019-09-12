import os
import numpy as np
import tensorflow as tf

from common_flags import COMMON_FLAGS

def run_validation(sess, right_count_op_list, placeholder_list, batch_size, dataset, istraining=False):
	'''
	Validation during training.
	Validation can be run on any set: training, validating or testing.

	Args:
		sess: run oprations in this session.
		input_op_list: list. For example, when validating on training set, it is [tr_videos_op, tr_action_labels_op, tr_actor_labels_op]
		right_count_op_list = [right_count_utility_op] + right_count_budget_op
		placeholder_list=[videos_placeholder, utility_labels_placeholder, budget_labels_placeholder, dropout_placeholder, istraining_placeholder]
		batch_size: cfg['TEST']['BATCH_SIZE']*GPU_NUM for testing and cfg['TRAIN']['BATCH_SIZE']*GPU_NUM for validation during training.
		dataset: string. 'val' or 'test'  
	Returns:
		test_correct_num_lst: result of right_count_op_list
        test_acc_lst: test_correct_num_lst./total_v
        total_v: int. total number of testing/val set
	'''
	assert(dataset in ['val', 'test'])
	# istraining = True
	# istraining = False

	test_videos = np.load(os.path.join(COMMON_FLAGS.FILES_NPY_DIR, '%s_videos.npy' % dataset))
	test_action_labels = np.load(os.path.join(COMMON_FLAGS.FILES_NPY_DIR, '%s_action_labels.npy' % dataset))
	test_actor_labels = np.load(os.path.join(COMMON_FLAGS.FILES_NPY_DIR, '%s_actor_labels.npy' % dataset))

	
	_op_num = len(right_count_op_list)
	test_correct_num_lst = [0.0] * _op_num

	total_v = test_videos.shape[0] # total number of testing samples

	batch_num = total_v // batch_size
	res_num = total_v % batch_size

	c = 0
	flag = 0
	while(flag==0):
		start_idx = c * batch_size
		end_idx = (c+1) * batch_size
		c += 1
		if end_idx == total_v:
			assert(res_num==0 and c==batch_num)
			flag = 1
		elif end_idx > total_v:
			assert(res_num!=0 and c==batch_num+1)
			flag = 2
			end_idx = total_v

		# get batch:
		test_videos_batch = test_videos[start_idx:end_idx, :, :, :, :]
		test_actor_labels_batch = test_actor_labels[start_idx:end_idx]
		test_action_labels_batch = test_action_labels[start_idx:end_idx]

		# padding:
		if flag == 2:
			# print('padding')
			test_videos_batch = np.pad(test_videos_batch, ((0,batch_size-res_num),(0,0),(0,0),(0,0),(0,0)), 'constant', constant_values=0)
			test_actor_labels_batch = np.pad(test_actor_labels_batch, ((0,batch_size-res_num)), 'constant', constant_values=-1)
			test_action_labels_batch = np.pad(test_action_labels_batch, ((0,batch_size-res_num)), 'constant', constant_values=-1)
		# the padded videos will never be true, since it can never be classified as -1
		if dataset in ['test']:
			print('test_videos_batch:', test_videos_batch.shape)
			print('test_actor_labels_batch:', test_actor_labels_batch.shape)
			print('test_action_labels_batch:', test_action_labels_batch.shape)
		# feed dorward:
		right_counts = sess.run(right_count_op_list, 
			feed_dict={placeholder: ndarray 
				for placeholder, ndarray in zip(
					placeholder_list, 
					[test_videos_batch, test_action_labels_batch, test_actor_labels_batch, 1.0, istraining])},
			options = tf.RunOptions(report_tensor_allocations_upon_oom = True)
			)
		if dataset in ['test']:
			print('right_counts:', right_counts)
		assert(len(right_counts) == _op_num)

		for i in range(_op_num):
			test_correct_num_lst[i] += right_counts[i]
	
	test_acc_lst = [_test_correct_num/total_v for _test_correct_num in test_correct_num_lst]

	return test_correct_num_lst, test_acc_lst, total_v