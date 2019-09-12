import os

home_dir = "/home/wuzhenyu_sjtu/Privacy-AdversarialLearning/data-collection/SBU"

filenames = []
videos_dir = os.path.join(home_dir, "videos/RGB")
nparts=5

for path, subdirs, files in os.walk(videos_dir):
    for name in files:
        filenames.append(os.path.join(path, name))

size_dir = []
for file in filenames:
    size_dir.append((file.split('/')[-1], file, os.path.getsize(file)))
    #print((file.split('/')[-1], file, os.path.getsize(file)))

import operator
size_dir.sort(key=operator.itemgetter(2))
#print(size_dir)

lstoflst = []
for i in range(nparts):
    lstoflst.append(size_dir[i::nparts])
print(len(lstoflst))
print(sum(len(lst) for lst in lstoflst))

for lst in lstoflst:
    print(sum(pair[2] for pair in lst))

import shutil
for i in range(nparts):
    newdir = os.path.join(home_dir, 'videos/RGB/part{}'.format(i))
    if not os.path.exists(newdir):
        os.makedirs(newdir)
    for file in lstoflst[i]:
        shutil.move(file[1], os.path.join(newdir, file[0]))
