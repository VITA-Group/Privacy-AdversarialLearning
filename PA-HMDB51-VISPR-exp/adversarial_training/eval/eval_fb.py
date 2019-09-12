import sys, time, os, datetime, errno, pprint, itertools
from os.path import dirname, realpath
sys.path.append(dirname(dirname(dirname(realpath(__file__)))))
sys.path.append(dirname(dirname(dirname(dirname(realpath(__file__))))))

import tensorflow as tf
from common_flags import COMMON_FLAGS
from eval_flags import FLAGS
from modules.degradNet import fd
from loss import *
from utils import *
from nets import nets_factory
from sklearn.metrics import average_precision_score
from data_preparation.VISPR.utils import *

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ["CUDA_VISIBLE_DEVICES"]=FLAGS.GPU_id

def create_architecture_fb(scope, loss_fb_lst_dct, logits_fb_lst_dct, fb_dict, batch_size, images, labels):
    with tf.variable_scope(tf.get_variable_scope(), reuse=False):
        fd_images = fd(images)
    loss_fb_images = 0.0
    logits_fb_images = tf.zeros([batch_size, COMMON_FLAGS.NUM_CLASSES_BUDGET])
    for name, fb in fb_dict.items():
        print(name)
        with tf.variable_scope(tf.get_variable_scope(), reuse=False):
            logits, _ = fb(fd_images)
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
        logits_fb_images += logits
        logits_fb_lst_dct[name].append(logits)
        loss_fb_lst_dct[name].append(loss)
        loss_fb_images += loss
    loss_fb_images = tf.divide(loss_fb_images, 4.0, 'LossFbMean')
    logits_fb_images = tf.divide(logits_fb_images, 4.0, 'LogitsFbMean')
    return loss_fb_images, logits_fb_images


def build_graph(gpu_num, batch_size, is_training):
    graph = tf.Graph()
    with graph.as_default():
        images_placeholder = tf.placeholder(tf.float32, shape=(batch_size * gpu_num, COMMON_FLAGS.CROP_HEIGHT, COMMON_FLAGS.CROP_WIDTH, COMMON_FLAGS.NCHANNEL))
        labels_placeholder = tf.placeholder(tf.float32, shape=(batch_size * gpu_num, COMMON_FLAGS.NUM_CLASSES_BUDGET))
        isTraining_placeholder = tf.placeholder(tf.bool)

        from collections import defaultdict
        logits_fb_lst_dct = defaultdict(list)
        loss_fb_lst_dct = defaultdict(list)

        logits_fb_lst = []
        loss_fb_lst = []

        fb_name_dict = {4: ['resnet_v1_50', 'resnet_v2_50', 'mobilenet_v1', 'mobilenet_v1_075'],
                        3: ['resnet_v1_50', 'resnet_v2_50', 'mobilenet_v1'],
                        2: ['resnet_v1_50', 'resnet_v2_50'],
                        1: ['resnet_v1_50']}
        fb_name_lst = fb_name_dict[FLAGS.M]

        fb_dict = {}
        for model_name in fb_name_lst:
            fb_dict[model_name] = nets_factory.get_network_fn(model_name,
                                                              num_classes=COMMON_FLAGS.NUM_CLASSES_BUDGET,
                                                              weight_decay=FLAGS.weight_decay,
                                                              is_training=isTraining_placeholder)
        with tf.variable_scope(tf.get_variable_scope()) as scope:
            for gpu_index in range(0, FLAGS.GPU_NUM):
                with tf.device('/gpu:%d' % gpu_index):
                    print('/gpu:%d' % gpu_index)
                    with tf.name_scope('%s_%d' % ('gpu', gpu_index)) as scope:
                        images = images_placeholder[gpu_index * batch_size:(gpu_index + 1) * batch_size]
                        labels = labels_placeholder[gpu_index * batch_size:(gpu_index + 1) * batch_size, :]
                        loss_fb, logits_fb = create_architecture_fb(scope, loss_fb_lst_dct, logits_fb_lst_dct, fb_dict, batch_size, images, labels)
                        loss_fb_lst.append(loss_fb)
                        logits_fb_lst.append(logits_fb)

                        tf.get_variable_scope().reuse_variables()

        loss_fb_op = tf.reduce_mean(loss_fb_lst, name='softmax')

        logits_fb_op = tf.concat(logits_fb_lst, axis=0)

        logits_fb_op_lst = []
        for model_name in fb_name_lst:
            logits_fb_op_lst.append(tf.concat(logits_fb_lst_dct[model_name], axis=0))

        if is_training:
            images_op, labels_op = create_images_reading_ops(is_train=True, is_val=False, GPU_NUM=gpu_num, BATCH_SIZE=batch_size, NUM_EPOCHS=1)
        else:
            images_op, labels_op = create_images_reading_ops(is_train=False, is_val=False, GPU_NUM=gpu_num, BATCH_SIZE=batch_size, NUM_EPOCHS=1)

        init_op = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())
        return (graph, init_op, images_placeholder, labels_placeholder, isTraining_placeholder, loss_fb_op, logits_fb_op,
                logits_fb_op_lst, images_op, labels_op, fb_name_lst)

def run_testing_fb():
    dir_path = FLAGS.adv_ckpt_dir
    ckpt_files = [".".join(f.split(".")[:-1]) for f in os.listdir(dir_path) if os.path.isfile(
        os.path.join(dir_path, f)) and '.data' in f]

    for ckpt_file in ckpt_files:
        for is_training in [True, False]:
            config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
            config.gpu_options.allow_growth = True
            tf.reset_default_graph()
            (graph, init_op, images_placeholder, labels_placeholder, isTraining_placeholder, loss_fb_op, logits_fb_op,
             logits_fb_op_lst, images_op, labels_op, fb_name_lst) = build_graph(FLAGS.GPU_NUM, FLAGS.image_batch_size, is_training)
            with tf.Session(graph=graph, config=config) as sess:
                sess.run(init_op)

                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=sess, coord=coord)

                saver = tf.train.Saver(tf.trainable_variables())

                saver.restore(sess, os.path.join(dir_path, ckpt_file))
                print('Session restored from trained model at {}!'.format(os.path.join(dir_path, ckpt_file)))

                fb_name_lst += ['ensemble']
                loss_budget_lst = []
                pred_probs_lst_lst = [[] for _ in range(len(fb_name_lst))]
                gt_lst = []
                try:
                    while not coord.should_stop():
                        images, labels = sess.run(
                                [images_op, labels_op])
                        gt_lst.append(labels)
                        value_lst = sess.run([loss_fb_op, logits_fb_op] + logits_fb_op_lst,
                                    feed_dict={images_placeholder: images,
                                               labels_placeholder: labels,
                                               isTraining_placeholder: True})
                        print(labels.shape)
                        loss_budget_lst.append(value_lst[0])
                        for i in range(len(fb_name_lst)):
                            pred_probs_lst_lst[i].append(value_lst[i+1])
                except tf.errors.OutOfRangeError:
                    print('Done testing on all the examples')
                finally:
                    coord.request_stop()
                coord.join(threads)

                gt_mat = np.concatenate(gt_lst, axis=0)
                n_examples, n_labels = gt_mat.shape
                for i in range(len(fb_name_lst)):
                    save_dir = os.path.join(FLAGS.adv_ckpt_dir, ckpt_file.split('.')[-1])
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    isTraining = lambda bool: "training" if bool else "testing"
                    with open(os.path.join(save_dir, '{}_class_scores_{}.txt'.format(fb_name_lst[i], isTraining(is_training))), 'w') as wf:
                        pred_probs_mat = np.concatenate(pred_probs_lst_lst[i], axis=0)
                        wf.write('# Examples = {}\n'.format(n_examples))
                        wf.write('# Labels = {}\n'.format(n_labels))
                        wf.write('Average Loss = {}\n'.format(np.mean(loss_budget_lst)))
                        wf.write("Macro MAP = {:.2f}\n".format(100 * average_precision_score(gt_mat, pred_probs_mat, average='macro')))
                        cmap_stats = average_precision_score(gt_mat, pred_probs_mat, average=None)
                        attr_id_to_name, attr_id_to_idx = load_attributes('../../../data_preparation/PA_HMDB51/attributes_pa_hmdb51.csv')
                        idx_to_attr_id = {v: k for k, v in attr_id_to_idx.items()}
                        wf.write('\t'.join(['attribute_id', 'attribute_name', 'num_occurrences', 'ap']) + '\n')
                        for idx in range(n_labels):
                            attr_id = idx_to_attr_id[idx]
                            attr_name = attr_id_to_name[attr_id]
                            attr_occurrences = np.sum(gt_mat, axis=0)[idx]
                            ap = cmap_stats[idx]
                            wf.write('{}\t{}\t{}\t{}\n'.format(attr_id, attr_name, attr_occurrences, ap * 100.0))

if __name__ == '__main__':
    run_testing_fb()