# coding: utf-8

from os import listdir,environ
from sys import argv
from os.path import isfile,join
import os


filenames = []
videos_dir = "hmdb51_orig"
nparts=500

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
if not os.path.exists('hmdb51_shuffled'):
    os.makedirs('hmdb51_shuffled')
for i in range(nparts):
    newdir = 'hmdb51_shuffled/part{}'.format(i)
    if not os.path.exists(newdir):
        os.makedirs(newdir)
    for file in lstoflst[i]:
        shutil.copyfile(file[1], os.path.join(newdir, file[0]))
