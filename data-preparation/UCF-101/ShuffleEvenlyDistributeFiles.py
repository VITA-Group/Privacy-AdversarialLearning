# coding: utf-8

from os import listdir,environ
from sys import argv
from os.path import isfile,join
import os
import argparse
import shutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, default='UCF-101', help='Directory of the source videos')
    parser.add_argument('--dst', type=str, default='UCF-101_shuffled', help='Directory of the destination videos')
    parser.add_argument('--nfolders', type=int, default=500, help='Number of output folders')
    args = parser.parse_args()
    params = vars(args)
    src_videos_dir = params['src']
    dst_videos_dir = params['dst']
    nfolders = params['nfolders']

    if not os.path.exists(dst_videos_dir):
        os.makedirs(dst_videos_dir)

    filenames = []
    for path, subdirs, files in os.walk(src_videos_dir):
        for name in files:
            filenames.append(os.path.join(path, name))

    lstOfTuples = []
    for filename in filenames:
        lstOfTuples.append((filename.split('/')[-1], filename, os.path.getsize(filename)))
        #print((file.split('/')[-1], file, os.path.getsize(file)))

    import operator
    # Sort the list of tuples according the size of file (3rd element in the tuple)
    lstOfTuples.sort(key=operator.itemgetter(2))

    lstOfLsts = []
    for i in range(nfolders):
        lstOfLsts.append(lstOfTuples[i::nfolders])

    for lst in lstOfLsts:
        print(sum(pair[2] for pair in lst))

    for i in range(nfolders):
        dst_dir = '{}/folder{}'.format(dst_videos_dir, i)
        if not os.path.exists(dst_dir):
            os.makedirs(newdir)
        for filename in lstOfLsts[i]:
            shutil.move(filename[1], os.path.join(dst_dir, filename[0]))

if __name__ == '__main__':
    main()
