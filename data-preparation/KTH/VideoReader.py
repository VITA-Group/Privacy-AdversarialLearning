import numpy as np
import cv2
import os
from tqdm import tqdm
import copy
from matlab_imresize import imresize

class VideoReader:
    def __init__(self, depth, height, width, sigma=None, ksize=None):
        self.width = width
        self.height = height
        self.depth = depth
        self.sigma = sigma
        self.ksize = ksize

    def read(self, filename, skip=True):
        cap = cv2.VideoCapture(filename)
        nframe = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(nframe)
        if skip:
            if nframe // self.depth >= 2:
                skip_frames = nframe % self.depth + self.depth
            else:
                skip_frames = nframe % self.depth
        else:
            skip_frames = nframe % self.depth
        start = skip_frames//2
        end = nframe-skip_frames//2
        if skip_frames % 2 == 1:
            end -= 1
        framelst = []
        for i in range(start, end):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            framelst.append(frame)
        cap.release()
        return np.split(np.array(framelst), int(nframe-skip_frames) // self.depth, axis=0)

    def read_gblur(self, filename, skip=True):
        print('Gaussian Blurring')
        cap = cv2.VideoCapture(filename)
        nframe = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(nframe)
        if skip:
            if nframe // self.depth >= 2:
                skip_frames = nframe % self.depth + self.depth
            else:
                skip_frames = nframe % self.depth
        else:
            skip_frames = nframe % self.depth
        start = skip_frames//2
        end = nframe-skip_frames//2
        if skip_frames % 2 == 1:
            end -= 1
        framelst = []
        for i in range(start, end):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = self.gaussian_blur(frame, self.ksize, self.sigma)
            subimage_arr = np.array(self.get_subimages(frame))
            framelst.append(subimage_arr)
        cap.release()
        print(np.array(framelst).shape)
        clips_lst = np.split(np.array(framelst), (nframe-skip_frames) // self.depth, axis=0)
        clip_lst = []
        for clips in clips_lst:
            clip_lst += [np.squeeze(clip, axis=1) for clip in np.split(clips, self.ksize*self.ksize, axis=1)]
        return clip_lst

    def read_avgblur(self, filename, skip=True):
        print('Average Blurring')
        cap = cv2.VideoCapture(filename)
        nframe = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(nframe)
        if skip:
            if nframe // self.depth >= 2:
                skip_frames = nframe % self.depth + self.depth
            else:
                skip_frames = nframe % self.depth
        else:
            skip_frames = nframe % self.depth
        start = skip_frames//2
        end = nframe-skip_frames//2
        if skip_frames % 2 == 1:
            end -= 1
        framelst = []
        for i in range(start, end):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = self.dsampleWithAvg(frame, self.ksize)
            frame = imresize(frame, output_shape=(120, 160))
            framelst.append(frame)
        cap.release()
        print(np.array(framelst).shape)
        print(nframe-skip_frames)
        return np.split(np.array(framelst), (nframe-skip_frames) // self.depth, axis=0)

    def get_subimages(self, frame):
        subimage_lst = []
        for x in range(self.ksize):
            for y in range(self.ksize):
                mask = np.fromfunction(lambda i, j, k: (i%self.ksize==x) & (j%self.ksize==y), frame.shape, dtype=int)
                subimage = frame[mask].reshape(frame.shape[0]//self.ksize, frame.shape[1]//self.ksize, -1)
                subimage = imresize(subimage, output_shape=(120, 160))
                subimage_lst.append(subimage)
        return subimage_lst

    def get_kth_action(self, filename):
        action_dict = {
            'running':0,
            'walking':1,
            'jogging':2,
            'handwaving':3,
            'handclapping':4,
            'boxing':5
        }
        action_key = filename.split('_')[1]
        return action_dict[action_key]

    def get_kth_actor(self, filename):
        actor_dict = {
        'person01':0,
        'person02':1,
        'person03':2,
        'person04':3,
        'person05':4,
        'person06':5,
        'person07':6,
        'person08':7,
        'person09':8,
        'person10':9,
        'person11':10,
        'person12':11,
        'person13':12,
        'person14':13,
        'person15':14,
        'person16':15,
        'person17':16,
        'person18':17,
        'person19':18,
        'person20':19,
        'person21':20,
        'person22':21,
        'person23':22,
        'person24':23,
        'person25':24,
        }
        #print(filename)
        #print(filename.index('_'))
        actor_key = filename.split('_')[0]
        return actor_dict[actor_key]
    # Data augmentation
    def loaddata_LR(self, video_dir):
        #print('start loading data')
        #files = [(root, file) for root, dir, files in os.walk(video_dir) for file in files]
        files = os.listdir(video_dir)
        #print(files)

        videos = []
        action_labels = []
        actor_labels = []
        pbar = tqdm(total=len(files))

        for filename in files:
            pbar.update(1)
            if filename == '.DS_Store':
                continue
            print(filename)
            #name = os.path.join(root, filename)
            try:
                #clips = self.read_gblur(name)
                action = filename.split('_')[1]
                skip = action in ['running', 'walking', 'jogging']
                filepath =  os.path.join(video_dir, filename)
                if 'train' in video_dir:
                    clips = self.read_gblur(filepath, skip)
                else:
                    clips = self.read_avgblur(filepath, skip)
                videos.extend(clips)
                action_label = self.get_kth_action(filename)
                actor_label = self.get_kth_actor(filename)
                action_labels.extend([action_label]*len(clips))
                actor_labels.extend([actor_label]*len(clips))
            except cv2.error as e:
                print(e)
                print('{} is unable to read'.format(filename))
                pass
        pbar.close()

        print(action_labels)
        print(actor_labels)

        return np.asarray(videos).astype('uint8'), np.asarray(action_labels).astype('uint32'), np.asarray(actor_labels).astype('uint32')

    def loaddata_HR(self, video_dir):
        #print('start loading data')
        #files = [(root, file) for root, dir, files in os.walk(video_dir) for file in files]
        files = os.listdir(video_dir)
        #print(files)

        videos = []
        action_labels = []
        actor_labels = []
        pbar = tqdm(total=len(files))

        for filename in files:
            pbar.update(1)
            if filename == '.DS_Store':
                continue
            print(filename)
            #name = os.path.join(root, filename)
            try:
                action = filename.split('_')[1]
                skip = action in ['running', 'walking', 'jogging']
                filepath =  os.path.join(video_dir, filename)
                clips = self.read(filepath, skip)
                videos.extend(clips)
                action_label = self.get_kth_action(filename)
                actor_label = self.get_kth_actor(filename)
                action_labels.extend([action_label]*len(clips))
                actor_labels.extend([actor_label]*len(clips))
            except cv2.error as e:
                print(e)
                print('{} is unable to read'.format(filename))
                pass
        pbar.close()

        print(action_labels)
        print(actor_labels)

        return np.asarray(videos).astype('uint8'), np.asarray(action_labels).astype('uint32'), np.asarray(actor_labels).astype('uint32')

    def loaddata_SR(self, video_dir):
        #print('start loading data')
        #files = [(root, file) for root, dir, files in os.walk(video_dir) for file in files]
        files = os.listdir(video_dir)
        #print(files)

        videos_LR = []
        videos_HR = []
        pbar = tqdm(total=len(files))

        for filename in files:
            pbar.update(1)
            if filename == '.DS_Store':
                continue
            print(filename)
            #name = os.path.join(root, filename)
            try:
                #clips = self.read_gblur(name)
                action = filename.split('_')[1]
                skip = action in ['running', 'walking', 'jogging']
                filepath =  os.path.join(video_dir, filename)
                clips_LR = self.read_avgblur(filepath, skip)
                clips_HR = self.read(filepath, skip)
                videos_LR.extend(clips_LR)
                videos_HR.extend(clips_HR)

            except cv2.error as e:
                print(e)
                print('{} is unable to read'.format(filename))
                pass
        pbar.close()

        return np.asarray(videos_LR).astype('uint8'), np.asarray(videos_HR).astype('uint8')

    def gaussian_kernel(self, sigma, ksize):
        radius = (ksize - 1) / 2.0
        x, y = np.mgrid[-radius:radius + 1, -radius:radius + 1]
        sigma = sigma ** 2
        k = 2 * np.exp(-0.5 * (x ** 2 + y ** 2) / sigma)
        k = k / np.sum(k)
        k = np.expand_dims(k, axis=2)
        return np.tile(k, (1,1,3))

    def tile_and_reflect(self, input):
        tiled_input = np.tile(input, (3, 3, 1))
        rows = input.shape[0]
        cols = input.shape[1]

        for i in range(3):
            tiled_input[i*rows:(i+1)*rows, 0:cols, :] = np.fliplr(tiled_input[i*rows:(i+1)*rows, 0:cols, :])
            tiled_input[i*rows:(i+1)*rows, -cols:, :] = np.fliplr(tiled_input[i*rows:(i+1)*rows, -cols:, :])

        for i in range(3):
            tiled_input[0:rows, i*cols:(i+1)*cols, :] = np.flipud(tiled_input[0:rows, i*cols:(i+1)*cols, :])
            tiled_input[-rows:, i*cols:(i+1)*cols, :] = np.flipud(tiled_input[-rows:, i*cols:(i+1)*cols, :])

        assert(np.array_equal(input, tiled_input[rows:2*rows, cols:2*cols, :]))

        assert(np.array_equal(input[0, :, :], tiled_input[rows-1, cols:2*cols, :]))
        assert(np.array_equal(input[:, -1, :], tiled_input[rows:2*rows, 2*cols, :]))
        assert(np.array_equal(input[-1, :, :], tiled_input[2*rows, cols:2*cols, :]))
        assert(np.array_equal(input[:, 0, :], tiled_input[rows:2*rows, cols-1, :]))

        return tiled_input

    def convolve(self, input, weights):
        assert(len(input.shape) == 3)
        assert(len(weights.shape) == 3)

        assert(weights.shape[0] < input.shape[0] + 1)
        assert(weights.shape[0] < input.shape[1] + 1)

        output = np.copy(input)
        tiled_input = self.tile_and_reflect(input)

        rows = input.shape[0]
        cols = input.shape[1]
        hw_row = weights.shape[0] // 2
        hw_col = weights.shape[1] // 2

        for i, io in zip(range(rows, rows*2), range(rows)):
            for j, jo in zip(range(cols, cols*2), range(cols)):
                average = 0.0
                overlapping = tiled_input[i-hw_row:i+hw_row,
                                        j-hw_col:j+hw_col, :]
                assert(overlapping.shape == weights.shape)
                tmp_weights = weights
                merged = tmp_weights[:] * overlapping
                average = np.sum(merged, axis=(0,1))
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
            img = image[:,:,i]
            blocks = self.extract_blocks(img, (ksize, ksize))
            lst = []
            for block in blocks:
                lst.append(np.mean(block))
            ds.append(np.array(lst).reshape(int(img.shape[0] / ksize), int(img.shape[1] / ksize)))
        return np.transpose(np.array(ds), (1,2,0))

    def extract_blocks(self, img, blocksize):
        M, N = img.shape
        b0, b1 = blocksize
        return img.reshape(M // b0, b0, N // b1, b1).swapaxes(1, 2).reshape(-1, b0, b1)
