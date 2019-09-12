
import os

filenames = []
class_dict = {}
video_dir = '../hmdb51_orig'
i = 0
for path, subdirs, files in os.walk(video_dir):
    if files:
        class_dict[path.split('/')[-1]] = i
        i += 1
    for name in files:
        filenames.append(os.path.join(path,name))

video_class_dict = {}
for path, subdirs, files in os.walk(video_dir):
    for name in files:
        video_class_dict[name] = path.split('/')[-1]

train_lst = []
test_lst = []
for file in os.listdir('test_set'):
    with open('test_set/'+file, "r") as f:
        for line in f:
            name = line.rstrip()
            test_lst.append(name)

with open('classInd.txt', 'w') as f:
    for k,v in class_dict.items():
        f.write(str(v+1)+'\t'+k+'\n')

with open('trainlist01.txt','w') as f:
    for name in video_class_dict.keys():
        if name not in test_lst:
            f.write(video_class_dict[name] + '/' + name + '\n')

with open('testlist01.txt', 'w') as f:
    for name in test_lst:
        f.write(video_class_dict[name] + '/' + name + '\n')

