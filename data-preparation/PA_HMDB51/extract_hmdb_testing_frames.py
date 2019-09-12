import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import os
from tqdm import tqdm
from random import randint
from scipy.misc import imsave

class VideoReader:
    def __init__(self):
        self.test_set = set()

    def read_test_split(self, filename_test):
        print(filename_test)
        with open(filename_test, 'r') as f:
            for line in f:
                print(line)
                name = line.strip('\n').split('/')[-1]
                self.test_set.add(name)
        print(self.test_set)

    def read(self, filename):
        cap = cv2.VideoCapture(filename)
        nframe = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # We skip the last frame
        # The last frame of some videos is unable to be captured
        dst = os.path.join('pa_hmdb51_frames')
        print(dst)
        if not os.path.exists(dst):
            os.makedirs(dst)
        for i in range(nframe):
            try:
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                print(ret)
                #print(frame)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            except cv2.error as e:
                print(e)
                print('{} (frame {}) is unable to read'.format(filename, i))
                continue
            print(frame.shape)
            imsave(os.path.join(dst, '{}_{}_{}.png'.format(filename.split('/')[-2], filename.split('/')[-1].split('.')[0], i)), frame)
        cap.release()

    def read_videos(self, video_dir):
        filenames = []
        #path_set = set([])
        for path, subdirs, files in os.walk(video_dir):
            for name in files:
                #if path in path_set:
                #    #print(path)
                #    continue
                #else:
                if name in self.test_set:
                    filenames.append(os.path.join(path, name))
                #path_set.add(path)
        print(filenames)
        pbar = tqdm(total=len(filenames))
        for filename in filenames:
            pbar.update(1)
            self.read(filename)

if __name__ == "__main__":
    vreader = VideoReader()
    vreader.read_test_split('privacy_split/testlist01.txt')
    vreader.read_videos('../HMDB51/hmdb51_orig')
