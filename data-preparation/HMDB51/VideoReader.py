import numpy as np
import cv2
import os
from tqdm import tqdm

class VideoReader:

    def __init__(self, depth, height, width, color, factor):
        self.width = width
        self.height = height
        self.depth = depth
        self.color = color
        self.factor = factor
        self.label_dict = {}
        self.train_set = set()
        self.test_set = set()
        self.name_class_dict = {}

    def read_ucf_labels(self, filename):
        with open(filename, 'r') as f:
            for line in f:
                words = line.strip('\n').split()
                self.label_dict[words[1].lower()] = int(words[0]) - 1

    def read_train_test_split(self, filename_train, filename_test):
        with open(filename_train, 'r') as f:
            for line in f:
                name = line.strip('\n').split('/')[-1]
                self.train_set.add(name)
                self.name_class_dict[name] = line.strip('\n').split('/')[0]
        with open(filename_test, 'r') as f:
            for line in f:
                name = line.strip('\n').split('/')[-1]
                self.test_set.add(name)
                self.name_class_dict[name] = line.strip('\n').split('/')[0]

    def get_hmdb_classname(self, fname):
        return self.label_dict[self.name_class_dict[fname]]

    def read(self, filename):
        cap = cv2.VideoCapture(filename)
        nframe = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        framearray = []
        print(nframe)
        nframe = nframe - nframe % self.depth
        print(nframe)
        for i in range(nframe):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, i-1)
                ret, frame = cap.read()
            # Convert to RGB space
            # By default, opencv reads the image in BGR mode
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if not self.color:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                frame = np.expand_dims(frame, axis=2)
            #frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
            #frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_CUBIC)
            #frame = cv2.resize(frame, (int(self.width * self.factor), int(self.height * self.factor)), interpolation=cv2.INTER_CUBIC)
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_CUBIC)
            framearray.append(frame)
        cap.release()
        return np.split(np.array(framearray), nframe // self.depth)

    def loaddata(self, video_dir):
        with open('{}_err.txt'.format(video_dir), 'w') as f:
            filenames = [os.path.join(video_dir, f) for f in os.listdir(video_dir)]
            '''
            for path, subdirs, files in os.walk(video_dir):
                for name in files:
                    filenames.append(os.path.join(path, name))
            '''
            videos_train = []
            videos_test = []

            labels_train = []
            labels_test = []
            pbar = tqdm(total=len(filenames),)
            for filename in filenames:
                pbar.update(1)
                print(filename)
                try:
                    clips = self.read(filename)
                    fname = filename.split('/')[-1]
                    print(fname)
                    if fname in self.train_set:
                        label = self.get_hmdb_classname(fname)
                        print(label)
                        labels_train.extend([label]*len(clips))
                        videos_train.extend(clips)
                    elif fname in self.test_set:
                        label = self.get_hmdb_classname(fname)
                        print(label)
                        labels_test.extend([label]*len(clips))
                        videos_test.extend(clips)
                    else:
                        print('{} not found in train/test split'.format(fname))
                        continue
                except cv2.error as e:
                    print(e)
                    print('{} is unable to read'.format(filename))
                    f.write('{} is unable to read\n'.format(filename))
                    pass
            pbar.close()
            print(len(labels_train))
            print(len(labels_test))
        return (
            np.asarray(videos_train, dtype='uint8'), np.asarray(labels_train),
            np.asarray(videos_test, dtype='uint8'), np.asarray(labels_test),)
