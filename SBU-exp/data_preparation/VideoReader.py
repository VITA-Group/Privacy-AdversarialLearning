import cv2
import numpy as np
import os
from scipy import stats
from tqdm import tqdm

from utils.matlab_imresize import imresize


class SBUReader:
    def __init__(self, depth, sigma=None, ksize=None):
        self.depth = depth
        self.sigma = sigma
        self.ksize = ksize
        self.val_set = set()
        self.test_set = set()
        self.read_val_test_split('SBU_Kinect/split/val.split', 'SBU_Kinect/split/test.split')

    def read_val_test_split(self, filename_val, filename_test):
        with open(filename_val, 'r') as f:
            for line in f:
                self.val_set.add('_'.join(line.strip('\n').split('/')))
        with open(filename_test, 'r') as f:
            for line in f:
                self.test_set.add('_'.join(line.strip('\n').split('/')))

    def indices_augment(self, nframe, clip_len, ncandidate):
        indices_lst = []
        for i in range(ncandidate):
            indices = np.random.choice(nframe, clip_len, replace=False)
            indices.sort()
            p_value = stats.kstest(indices, stats.uniform(loc=0, scale=nframe).cdf)[1]
            indices_lst.append((indices, p_value))
        indices_lst.sort(key=lambda tup: tup[1], reverse=True)  # sorts in place
        indices = []
        for i in range(nframe // clip_len + 2):
            indices += indices_lst[i][0].tolist()
        return indices

    def read(self, filename):
        print('No Blurring')
        cap = cv2.VideoCapture(filename)
        nframe = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if nframe > self.depth:
            indices = self.indices_augment(nframe, self.depth, 20)
        else:
            indices = [x for x in range(nframe)] + [nframe-1] * (self.depth - nframe)
        framelst = []
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            print(frame.shape)
            #print(frame[:,:,0])
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = imresize(frame, output_shape=(120, 160))
            framelst.append(frame)
        cap.release()
        clip_lst = np.split(np.array(framelst), len(framelst) // self.depth, axis=0)
        return clip_lst

    def read_gblur(self, filename):
        print('Gaussian Blurring')
        cap = cv2.VideoCapture(filename)
        nframe = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if nframe > self.depth:
            indices = self.indices_augment(nframe, self.depth, 50)
        else:
            indices = [x for x in range(nframe)] + [nframe-1] * (self.depth - nframe)
        framelst = []
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = self.gaussian_blur(frame, self.ksize, self.sigma)
            subimage_arr = np.array(self.get_subimages(frame))
            framelst.append(subimage_arr)
        cap.release()
        clips_lst = np.split(np.array(framelst), len(framelst) // self.depth, axis=0)
        clip_lst = []
        for clips in clips_lst:
            clip_lst += [np.squeeze(clip, axis=1) for clip in np.split(clips, self.ksize * self.ksize, axis=1)]
        return clip_lst

    def read_avgblur(self, filename):
        print('Average Blurring')
        cap = cv2.VideoCapture(filename)
        nframe = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if nframe > self.depth:
            indices = self.indices_augment(nframe, self.depth, 50)
        else:
            indices = [x for x in range(nframe)] + [nframe-1] * (self.depth - nframe)
        framelst = []
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = self.dsampleWithAvg(frame, self.ksize)
            # frame = imresize(frame, output_shape=(120, 160))
            framelst.append(frame)
        cap.release()
        return np.split(np.array(framelst), len(framelst) // self.depth, axis=0)

    def get_subimages(self, frame):
        subimage_lst = []
        for x in range(self.ksize):
            for y in range(self.ksize):
                mask = np.fromfunction(lambda i, j, k: (i % self.ksize == x) & (j % self.ksize == y), frame.shape,
                                       dtype=int)
                subimage = frame[mask].reshape(frame.shape[0] // self.ksize, frame.shape[1] // self.ksize, -1)
                # subimage = imresize(subimage, output_shape=(120, 160))
                subimage_lst.append(subimage)
        return subimage_lst

    def get_SBU_action(self, path):
        action_dict = {
            '01': 0,
            '02': 1,
            '03': 2,
            '04': 3,
            '05': 4,
            '06': 5,
            '07': 6,
            '08': 7,
        }
        action_key = path.split('_')[1]
        return action_dict[action_key]

    def get_SBU_actor(self, path):
        actor_dict = {
            's01s02': 0,
            's01s03': 1,
            's01s07': 2,
            's02s01': 0,
            's02s03': 3,
            's02s06': 4,
            's02s07': 5,
            's03s02': 3,
            's03s04': 6,
            's03s05': 7,
            's03s06': 8,
            's04s02': 9,
            's04s03': 6,
            's04s06': 10,
            's05s02': 11,
            's05s03': 7,
            's06s03': 8,
            's06s02': 4,
            's06s04': 10,
            's07s01': 2,
            's07s03': 12,
        }
        # print(filename)
        # print(filename.index('_'))
        actor_key = path.split('_')[0]
        return actor_dict[actor_key]

    def get_SBU_setting(self, path):
        setting_dict = {
            '001': 0,
            '002': 1,
            '003': 2,
            '004': 3,
        }
        setting_key = path.split('_')[2]
        return setting_dict[setting_key]

    # Data augmentation
    def loaddata_LR(self, video_dir, train=True):
        videos = []
        action_labels = []
        actor_labels = []


        video_dir = os.path.normpath(video_dir)
        files = os.listdir(video_dir)
        pbar = tqdm(total=len(files))

        for filename in files:
            pbar.update(1)
            if filename == '.DS_Store':
                continue
            actor = self.get_SBU_actor(filename)
            action = self.get_SBU_action(filename)
            filepath = os.path.join(video_dir, filename)
            if train:
                clips = self.read_gblur(filepath)
                videos.extend(clips)
                action_labels.extend([action]*len(clips))
                actor_labels.extend([actor]*len(clips))
            else:
                clips = self.read_avgblur(filepath)
                videos.extend(clips)
                action_labels.extend([action]*len(clips))
                actor_labels.extend([actor]*len(clips))

        return np.asarray(videos).astype('uint8'), np.asarray(action_labels).astype('uint32'), \
               np.asarray(actor_labels).astype('uint32')

    def loaddata_HR(self, video_dir):
        train_videos = []
        train_action_labels = []
        train_actor_labels = []

        test_videos = []
        test_action_labels = []
        test_actor_labels = []

        val_videos = []
        val_action_labels = []
        val_actor_labels = []

        video_dir = os.path.normpath(video_dir)
        files = os.listdir(video_dir)
        pbar = tqdm(total=len(files))

        for filename in files:
            pbar.update(1)
            if filename == '.DS_Store':
                continue
            print(filename)
            print(self.val_set)
            print(self.test_set)
            actor = self.get_SBU_actor(filename)
            action = self.get_SBU_action(filename)
            filepath = os.path.join(video_dir, filename)
            clips = self.read(filepath)
            filename = filename.split('.')[0]
            if filename in self.val_set:
                print('validation')
                val_videos.extend(clips)
                val_action_labels.extend([action] * len(clips))
                val_actor_labels.extend([actor] * len(clips))

            elif filename in self.test_set:
                print('testing')
                test_videos.extend(clips)
                test_action_labels.extend([action] * len(clips))
                test_actor_labels.extend([actor] * len(clips))
            else:
                print('training')
                train_videos.extend(clips)
                train_action_labels.extend([action]*len(clips))
                train_actor_labels.extend([actor]*len(clips))
        train_lst = [np.asarray(train_videos).astype('uint8'), np.asarray(train_action_labels).astype('uint32'), \
               np.asarray(train_actor_labels).astype('uint32')]
        test_lst = [np.asarray(test_videos).astype('uint8'), np.asarray(test_action_labels).astype('uint32'), \
               np.asarray(test_actor_labels).astype('uint32')]
        val_lst = [np.asarray(val_videos).astype('uint8'), np.asarray(val_action_labels).astype('uint32'), \
               np.asarray(val_actor_labels).astype('uint32')]
        return train_lst, val_lst, test_lst

    def gaussian_kernel(self, sigma, ksize):
        radius = (ksize - 1) / 2.0
        x, y = np.mgrid[-radius:radius + 1, -radius:radius + 1]
        sigma = sigma ** 2
        k = 2 * np.exp(-0.5 * (x ** 2 + y ** 2) / sigma)
        k = k / np.sum(k)
        k = np.expand_dims(k, axis=2)
        return np.tile(k, (1, 1, 3))

    def tile_and_reflect(self, input):
        tiled_input = np.tile(input, (3, 3, 1))
        rows = input.shape[0]
        cols = input.shape[1]

        for i in range(3):
            tiled_input[i * rows:(i + 1) * rows, 0:cols, :] = np.fliplr(tiled_input[i * rows:(i + 1) * rows, 0:cols, :])
            tiled_input[i * rows:(i + 1) * rows, -cols:, :] = np.fliplr(tiled_input[i * rows:(i + 1) * rows, -cols:, :])

        for i in range(3):
            tiled_input[0:rows, i * cols:(i + 1) * cols, :] = np.flipud(tiled_input[0:rows, i * cols:(i + 1) * cols, :])
            tiled_input[-rows:, i * cols:(i + 1) * cols, :] = np.flipud(tiled_input[-rows:, i * cols:(i + 1) * cols, :])

        assert (np.array_equal(input, tiled_input[rows:2 * rows, cols:2 * cols, :]))

        assert (np.array_equal(input[0, :, :], tiled_input[rows - 1, cols:2 * cols, :]))
        assert (np.array_equal(input[:, -1, :], tiled_input[rows:2 * rows, 2 * cols, :]))
        assert (np.array_equal(input[-1, :, :], tiled_input[2 * rows, cols:2 * cols, :]))
        assert (np.array_equal(input[:, 0, :], tiled_input[rows:2 * rows, cols - 1, :]))

        return tiled_input

    def convolve(self, input, weights):
        assert (len(input.shape) == 3)
        assert (len(weights.shape) == 3)

        assert (weights.shape[0] < input.shape[0] + 1)
        assert (weights.shape[0] < input.shape[1] + 1)

        output = np.copy(input)
        tiled_input = self.tile_and_reflect(input)

        rows = input.shape[0]
        cols = input.shape[1]
        hw_row = weights.shape[0] // 2
        hw_col = weights.shape[1] // 2

        for i, io in zip(range(rows, rows * 2), range(rows)):
            for j, jo in zip(range(cols, cols * 2), range(cols)):
                average = 0.0
                overlapping = tiled_input[i - hw_row:i + hw_row,
                              j - hw_col:j + hw_col, :]
                assert (overlapping.shape == weights.shape)
                tmp_weights = weights
                merged = tmp_weights[:] * overlapping
                average = np.sum(merged, axis=(0, 1))
                output[io, jo, :] = average
        return output

    def gaussian_blur(self, img, ksize, sigma):
        k = self.gaussian_kernel(sigma, ksize)
        blurred_img = self.convolve(img, k)
        return blurred_img

    def dsampleWithAvg(self, image, ksize):
        ndim = 3
        ds = []
        for i in range(ndim):
            img = image[:, :, i]
            blocks = self.extract_blocks(img, (ksize, ksize))
            lst = []
            for block in blocks:
                lst.append(np.mean(block))
            ds.append(np.array(lst).reshape(int(img.shape[0] / ksize), int(img.shape[1] / ksize)))
        return np.transpose(np.array(ds), (1, 2, 0))

    def extract_blocks(self, img, blocksize):
        M, N = img.shape
        b0, b1 = blocksize
        return img.reshape(M // b0, b0, N // b1, b1).swapaxes(1, 2).reshape(-1, b0, b1)