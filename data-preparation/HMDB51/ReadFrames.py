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
        pass

    def read(self, filename):
        cap = cv2.VideoCapture(filename)
        nframe = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # We skip the last frame
        # The last frame of some videos is unable to be captured
        i = randint(0, nframe - 2)
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            #print(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cap.release()
        except cv2.error as e:
            print(e)
            print('{} is unable to read'.format(filename))
            return
        dst = os.path.join('hmdb51_frames', filename.split('/')[-2])
        print(dst)
        if not os.path.exists(dst):
            os.makedirs(dst)
        imsave('{}.png'.format(os.path.join(dst, filename.split('/')[-1].split('.')[0])), frame)

    def read_videos(self, video_dir):
        filenames = []
        path_set = set([])
        for path, subdirs, files in os.walk(video_dir):
            for name in files:
                #if path in path_set:
                #    #print(path)
                #    continue
                #else:
                print(os.path.join(path, name))
                filenames.append(os.path.join(path, name))
                #path_set.add(path)
        print(filenames)
        pbar = tqdm(total=len(filenames))
        for filename in filenames:
            pbar.update(1)
            self.read(filename)

if __name__ == "__main__":
    vreader = VideoReader()
    vreader.read_videos('hmdb51_orig')
