import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
import tensorflow as tf
from skimage import img_as_float
from skimage import exposure
import cv2
import sys
import yaml
import pprint
import os
import errno

sys.path.insert(0, '..')
from img_proc import _instance_norm, _binary_activation, _avg_replicate, _bilinear_resize
from degradlNet import residualNet
from input_data import *

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
#os.environ["CUDA_VISIBLE_DEVICES"]="0"

def linear_scaling(frame, vid, video_linear_scaling, joint_channels):
    if not video_linear_scaling:
        flst = []
        for j in range(frame.shape[2]):
            if joint_channels:
                max, min = np.max(frame), np.min(frame)
            else:
                max, min = np.max(frame[:, :, j]), np.min(frame[:, :, j])
            f = (frame[:, :, j] - min) / (max - min) * 255
            flst.append(f)
        frame = np.stack(flst, axis=2)
    else:
        flst = []
        for j in range(frame.shape[2]):
            if joint_channels:
                max, min = np.max(vid), np.min(vid)
            else:
                max, min = np.max(vid[:, :, :, j]), np.min(vid[:, :, :, j])
            f = (frame[:, :, j] - min) / (max - min) * 255
            flst.append(f)
        frame = np.stack(flst, axis=2)
    return frame

def plot_visualization_frame(eval_vis_dir, frame, name):
    def crop_center(img, cropx, cropy):
        y, x,_ = img.shape
        startx = x // 2 - (cropx // 2)
        starty = y // 2 - (cropy // 2)
        return img[starty:starty + cropy, startx:startx + cropx, :]

    def plot_img_and_hist(image, axes, bins=256):
        #Plot an image along with its histogram and cumulative histogram.
        image = img_as_float(image)
        ax_img, ax_hist = axes
        ax_cdf = ax_hist.twinx()

        # Display image
        ax_img.imshow(image, cmap=plt.cm.gray)
        ax_img.set_axis_off()
        ax_img.set_adjustable('box-forced')

        # Display histogram
        ax_hist.hist(image.ravel(), bins=bins, histtype='step', color='black')
        ax_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
        ax_hist.set_xlabel('Pixel intensity')
        ax_hist.set_xlim(0, 1)
        ax_hist.set_yticks([])

        # Display cumulative distribution
        img_cdf, bins = exposure.cumulative_distribution(image, bins)
        ax_cdf.plot(bins, img_cdf, 'r')
        ax_cdf.set_yticks([])

        return ax_img, ax_hist, ax_cdf

    frame = crop_center(frame, 112, 112)
    frame = frame[:,:,0]
    max, min = np.max(frame), np.min(frame)
    frame = ((frame - min) / (max - min) * 255).astype('uint8')
    # Contrast stretching
    p2, p98 = np.percentile(frame, (2, 98))
    img_rescale = exposure.rescale_intensity(frame, in_range=(p2, p98))
    # Equalization
    img_eq = exposure.equalize_hist(frame)

    # Display results
    fig = plt.figure(figsize=(12, 8))
    axes = np.zeros((2, 3), dtype=np.object)
    axes[0, 0] = fig.add_subplot(2, 3, 1)
    for i in range(1, 3):
        axes[0, i] = fig.add_subplot(2, 3, 1 + i, sharex=axes[0, 0], sharey=axes[0, 0])
    for i in range(3, 6):
        axes[i // 3, i % 3] = fig.add_subplot(2, 3, 1 + i)

    ax_img, ax_hist, ax_cdf = plot_img_and_hist(frame, axes[0:2, 0])
    ax_img.set_title('Feature map')

    y_min, y_max = ax_hist.get_ylim()
    ax_hist.set_ylabel('Number of pixels')
    ax_hist.set_yticks(np.linspace(0, y_max, 5))

    ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_rescale, axes[0:2, 1])
    ax_img.set_title('Contrast stretching')

    ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_eq, axes[0:2, 2])
    ax_img.set_title('Histogram equalization')


    ax_cdf.set_ylabel('Fraction of total intensity')
    ax_cdf.set_yticks(np.linspace(0, 1, 5))

    fig.tight_layout()

    plt.savefig(os.path.join(eval_vis_dir, '{}.png'.format(name)))
    plt.close()

def plot_visualization(vid, vid_orig, vis_dir, plot_img=False, write_video=True):
    def crop_center(img, cropx, cropy):
        y, x,_ = img.shape
        startx = x // 2 - (cropx // 2)
        starty = y // 2 - (cropy // 2)
        return img[starty:starty + cropy, startx:startx + cropx, :]

    def plot_img_and_hist(image, axes, bins=256):
        #Plot an image along with its histogram and cumulative histogram.
        image = img_as_float(image)
        ax_img, ax_hist = axes
        ax_cdf = ax_hist.twinx()

        # Display image
        ax_img.imshow(image, cmap=plt.cm.gray)
        ax_img.set_axis_off()
        ax_img.set_adjustable('box-forced')

        # Display histogram
        ax_hist.hist(image.ravel(), bins=bins, histtype='step', color='black')
        ax_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
        ax_hist.set_xlabel('Pixel intensity')
        ax_hist.set_xlim(0, 1)
        ax_hist.set_yticks([])

        # Display cumulative distribution
        img_cdf, bins = exposure.cumulative_distribution(image, bins)
        ax_cdf.plot(bins, img_cdf, 'r')
        ax_cdf.set_yticks([])

        return ax_img, ax_hist, ax_cdf

    def PCA(data):
        m, n = data.shape[0], data.shape[1]
        #print(m, n)
        mean = np.mean(data, axis=0)
        data -= np.tile(mean, (m, 1))
        # calculate the covariance matrix
        cov = np.matmul(np.transpose(data), data)
        evals, evecs = np.linalg.eigh(cov)
        # sort eigenvalue in decreasing order
        idx = np.argsort(evals)[::-1]
        evecs = evecs[:, idx]
        evals = evals[idx]
        #print(evals)
        evecs = evecs[:, 0]
        return np.matmul(data, evecs), evals[0] / sum(evals)

    width, height = 112, 112

    video_histeq = []
    for i in range(vid.shape[0]):
        frame = crop_center(vid[i], 112, 112)
        frame = np.reshape(frame, (112 * 112, 3))
        frame, K = PCA(frame)
        frame = np.reshape(frame, (112, 112))
        max, min = np.max(frame), np.min(frame)
        frame = ((frame - min) / (max - min) * 255).astype('uint8')
        # Contrast stretching
        p2, p98 = np.percentile(frame, (2, 98))
        img_rescale = exposure.rescale_intensity(frame, in_range=(p2, p98))
        # Equalization
        img_eq = exposure.equalize_hist(frame)
        video_histeq.append(img_eq)

        # # Adaptive Equalization
        # img_adapteq = exposure.equalize_adapthist(frame, clip_limit=0.03)

        if plot_img:
            # Display results
            fig = plt.figure(figsize=(12, 16))
            axes = np.zeros((4, 3), dtype=np.object)
            axes[0, 0] = fig.add_subplot(4, 3, 1)
            for j in range(1, 3):
                axes[0, j] = fig.add_subplot(4, 3, 1 + j, sharex=axes[0, 0], sharey=axes[0, 0])
            for j in range(3, 12):
                axes[j // 3, j % 3] = fig.add_subplot(4, 3, 1 + j)

            ax_img, ax_hist, ax_cdf = plot_img_and_hist(frame, axes[0:2, 0])
            ax_img.set_title('PCA on 3 channels ({:.4f})'.format(K))

            y_min, y_max = ax_hist.get_ylim()
            ax_hist.set_ylabel('Number of pixels')
            ax_hist.set_yticks(np.linspace(0, y_max, 5))

            ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_rescale, axes[0:2, 1])
            ax_img.set_title('Contrast stretching')

            ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_eq, axes[0:2, 2])
            ax_img.set_title('Histogram equalization')

            #ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_adapteq, axes[0:1, 3])
            #ax_img.set_title('Adaptive equalization')

            ax_cdf.set_ylabel('Fraction of total intensity')
            ax_cdf.set_yticks(np.linspace(0, 1, 5))

            print(vid_orig[j].shape)
            frame_downsample_crop = crop_center(vid_orig[j], 112, 112)
            frame = crop_center(vid[j], 112, 112)
            axes[2, 0].imshow(frame_downsample_crop.astype('uint8'))
            axes[2, 0].set_title('Dowmsampled')
            frame_scaled_joint = linear_scaling(frame, vid, video_linear_scaling=True, joint_channels=True).astype('uint8')
            axes[2, 1].imshow(frame_scaled_joint.astype('uint8'))
            axes[2, 1].set_title('Joint Scaling')
            frame_scaled_separate = linear_scaling(frame, vid, video_linear_scaling=True, joint_channels=False).astype('uint8')
            axes[2, 2].imshow(frame_scaled_separate.astype('uint8'))
            axes[2, 2].set_title('Separate Scaling')
            for j in range(frame.shape[2]):
                axes[3, j].imshow(frame[:,:,j], cmap=plt.get_cmap('jet'))
                axes[3, j].set_title('Channel{}'.format(j))
            # prevent overlap of y-axis labels
            fig.tight_layout()
            plt.savefig('{}/vis_{}.png'.format(vis_dir, i))
            plt.close()

    if write_video:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Be sure to use lower case
        output = "{}/hist_eq.avi".format(vis_dir)
        out = cv2.VideoWriter(output, fourcc, 10.0, (width, height), False)
        vid = np.multiply(np.asarray(video_histeq), 255).astype('uint8')
        print(vid.shape)
        print(output)
        for i in range(vid.shape[0]):
            frame = vid[i]
            #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            #frame = frame.reshape(112, 112, 3)
            # print(frame)
            out.write(frame)
        out.release()
        #cv2.destroyAllWindows()

def visualize(directory, X, X_orig, Y_action, Y_actor):
    def get_SBU_actor(actor_key):
        actor_dict = {
            0: 's01s02',
            1: 's01s03',
            2: 's01s07',
            3: 's02s03',
            4: 's02s06',
            5: 's02s07',
            6: 's03s04',
            7: 's03s05',
            8: 's03s06',
            9: 's04s02',
            10: 's04s06',
            11: 's05s02',
            12: 's07s03',
        }
        return actor_dict[actor_key]
    def get_SBU_action(action_key):
        action_dict = {
            0: '01',
            1: '02',
            2: '03',
            3: '04',
            4: '05',
            5: '06',
            6: '07',
            7: '08',
        }
        return action_dict[action_key]
    for i in range(len(X)):
        actor = get_SBU_actor(Y_actor[i])
        action = get_SBU_action(Y_action[i])
        vis_dir = os.path.join(directory, '{}_{}'.format(actor, action))
        if not os.path.exists(vis_dir):
            os.makedirs(vis_dir)
        else:
            continue
        vid = X[i]
        vid_orig = X_orig[i]
        plot_visualization(vid, vid_orig, vis_dir)

def visualize_degradation(checkpoint_dir):
    cfg = yaml.load(open('params.yml'))
    pp = pprint.PrettyPrinter()
    pp.pprint(cfg)
    if not os.path.exists(cfg['VIS_DIR']):
        os.makedirs(cfg['VIS_DIR'])

    videos_placeholder = tf.placeholder(tf.float32, shape=(cfg['BATCH_SIZE'] * cfg['GPU_NUM'], cfg['DEPTH'], 112, 112, cfg['NCHANNEL']))

    videos_degraded_lst = []
    with tf.variable_scope(tf.get_variable_scope()) as scope:
        for gpu_index in range(0, cfg['GPU_NUM']):
            with tf.device('/gpu:%d' % gpu_index):
                print('/gpu:%d' % gpu_index)
                with tf.name_scope('%s_%d' % ('gpu', gpu_index)) as scope:
                    videos_degraded = residualNet(
                        videos_placeholder[gpu_index * cfg['BATCH_SIZE']:(gpu_index + 1) * cfg['BATCH_SIZE']], is_video=True)
                    if cfg['USE_AVG_REPLICATE']:
                        videos_degraded = _avg_replicate(videos_degraded)
                    videos_degraded_lst.append(videos_degraded)

                    # Reuse variables for the next tower.
                    tf.get_variable_scope().reuse_variables()
    videos_degraded_op = tf.concat(videos_degraded_lst, 0)
    train_files = [os.path.join(cfg['TRAIN_FILES_DIR'], f) for
                   f in os.listdir(cfg['TRAIN_FILES_DIR']) if f.endswith('.tfrecords')]
    val_files = [os.path.join(cfg['VAL_FILES_DIR'], f) for
                 f in os.listdir(cfg['VAL_FILES_DIR']) if f.endswith('.tfrecords')]

    print(train_files)
    print(val_files)
    videos_op, action_labels_op, actor_labels_op = inputs_videos(filenames=val_files,
                                                                    batch_size=cfg['GPU_NUM'] * cfg['BATCH_SIZE'],
                                                                    num_epochs=1,
                                                                    num_threads=cfg['NUM_THREADS'],
                                                                    num_examples_per_epoch=cfg['NUM_EXAMPLES_PER_EPOCH'],
                                                                    shuffle=False)

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        saver = tf.train.Saver(tf.trainable_variables())
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir=checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Session restored from trained model at {}!'.format(ckpt.model_checkpoint_path))
        else:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), checkpoint_dir)
        try:
            #videos_degraded_lst = []
            #action_labels_lst = []
            #actor_labels_lst = []
            directory = cfg['VIS_DIR']
            while not coord.should_stop():
                videos, action_labels, actor_labels = sess.run([videos_op, action_labels_op, actor_labels_op])
                videos_degraded_value = sess.run(videos_degraded_op, feed_dict={videos_placeholder: videos})
                videos.tolist()
                videos_degraded_value.tolist()

                #videos_degraded_lst.append(videos_degraded_value*255)
                #videos_degraded_lst.extend(videos_degraded_value)
                #action_labels_lst.extend(action_labels)
                #actor_labels_lst.extend(actor_labels)

                visualize(directory, videos_degraded_value, videos, action_labels, actor_labels)
                #raise tf.errors.OutOfRangeError
        except tf.errors.OutOfRangeError:
            print('Done testing on all the examples')
        finally:
            coord.request_stop()
            coord.join(threads)

def main(_):
    checkpoint_dir = '/hdd1/wuzhenyu_sjtu/checkpoint_UseResidual/L1Loss_NoLambdaDecay_AvgReplicate_MonitorBudget_MonitorUtility_Resample_6_2.0_0.5'
    visualize_degradation(checkpoint_dir)


if __name__ == '__main__':
    tf.app.run()