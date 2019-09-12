import csv
import os
import cv2
import re

class Video:
    def __init__(self, actor, action, setting, label_lst):
        self.actor = actor
        self.action = action
        self.setting = setting
        self.label_lst = label_lst
        self.height = 120
        self.width = 160
    
    def segment(self):
        filename = "{}/{}_{}_{}_uncomp.avi".format("KTH", self.actor, self.action, self.setting)
        print(filename)
        if os.path.isfile(filename) and os.access(filename, os.R_OK):
            print("File exists and is readable")
        else:
            print("Either file is missing or is not readable")
        cap = cv2.VideoCapture(filename)
        index = 0
        for label in self.label_lst:
            index += 1
            label = label.split('-')
            start, end = int(label[0]), int(label[1])
            
            fourcc = cv2.VideoWriter_fourcc(*'XVID') # Be sure to use lower case
            if not os.path.exists(self.action):
                os.makedirs(self.action)
            if self.setting != 'd4':
                output = "{}/{}_{}_{}_{:d}.avi".format('train', self.actor, self.action, self.setting, index)
            else:
                output = "{}/{}_{}_{}_{:d}.avi".format('test', self.actor, self.action, self.setting, index)
            
            out = cv2.VideoWriter(output, fourcc, 30.0, (self.width, self.height), True)

            for i in range(start, end):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                #print(frame)
                if not ret:
                    print(output)
                    print(i)
                    continue
                out.write(frame)
            out.release()
            cv2.destroyAllWindows()
            print("The output video is {}".format(output))
        cap.release()

vid_lst = []
with open("00sequences.txt") as file:
    for line in file:
    #for line in csv.reader(tsv, delimiter="\t"):
        #if line:
        line = line.strip()
        line = re.sub(r" +", "", line)
        if line:
            line = line.split('\t')
            line = list(filter(None, line))
            actor, action, setting = line[0].split('_')
            label_lst = line[2].split(',')
            vid_lst.append(Video(actor, action, setting, label_lst))       

for vid in vid_lst:
    if vid.setting != 'd3':
        vid.segment()

