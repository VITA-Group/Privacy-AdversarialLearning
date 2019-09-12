#!/usr/bin/python

import json
import time
import pickle
import sys
import csv
import argparse
import os
import os.path as osp
import shutil

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from scipy.misc import imread

from vispr import DS_ROOT, CAFFE_ROOT
from vispr.tools.common.utils import *

sys.path.insert(0, os.path.join(CAFFE_ROOT, 'python'))
import caffe

sys.path.insert(1, CAFFE_ROOT + 'examples/pycaffe/layers')   # the datalayers we will use are in this directory.
sys.path.insert(1, CAFFE_ROOT + 'examples/pycaffe')   # the tools file is in this folder



def classify_paths(net, transformer, filenames, out_file, batch_size=64):
    n_files = len(filenames)

    net_img_size = net.blobs['data'].data.shape[-1]
    print '########################################################################################'
    print net_img_size
    print '########################################################################################'

    assert net_img_size in [224, 227, 321]

    # Perform Forward-Passes per image ---------------------------------------------------------------------------------
    for start_idx in range(0, len(filenames), batch_size):
        sys.stdout.write("Processing %d/%d (%.2f%% done) \r" % (start_idx + 1, n_files,
                                                                (start_idx + 1) * 100.0 / n_files))
        sys.stdout.flush()

        end_idx = min(start_idx + batch_size, len(filenames))
        this_batch_size = end_idx - start_idx
        this_batch = np.zeros((this_batch_size, 3, net_img_size, net_img_size))

        net.blobs['data'].reshape(this_batch_size,  # batch size
                                  3,  # 3-channel (BGR) images
                                  net_img_size, net_img_size)

        for idx, f_idx in enumerate(range(start_idx, end_idx)):
            image_path = filenames[f_idx]
            try:
                image = caffe.io.load_image(image_path)
                transformed_image = transformer.preprocess('data', image)
            except ValueError:
                print 'Unable to process: ', image_path
                print image.shape
            this_batch[idx] = transformed_image

        net.blobs['data'].data[...] = this_batch

        output = net.forward()
        output_probs = output['prob']

        with open(out_file, 'a') as wf:
            for idx, f_idx in enumerate(range(start_idx, end_idx)):
                image_path = filenames[f_idx]
                dct_entry = {'image_path': image_path, 'pred_probs': output_probs[idx].tolist()}
                json_str = json.dumps(dct_entry)
                wf.write(json_str + '\n')


def evaluate(params):
    def save_obj(obj, name):
        with open('obj/' + name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    def load_obj(name):
        with open('obj/' + name + '.pkl', 'rb') as f:
            return pickle.load(f)
    # Load Attributes --------------------------------------------------------------------------------------------------
    attr_id_to_name, attr_id_to_idx = load_attributes()
    idx_to_attr_id = {v: k for k, v in attr_id_to_idx.items()}
    attr_id_count = {k: 0 for k, v in attr_id_to_idx.items()}
    from collections import defaultdict
    attr_id_conf = defaultdict(list)

    # Load predictions -------------------------------------------------------------------------------------------------
    # Construct a list of dicts containing: GT labels, Prediction probabilities, Image path
    pred_list = []
    with open(params['pred_file'], 'r') as f:
        for _line in f:
            line = _line.strip()
            dct = json.loads(line)

            pred_entry = dict()
            pred_entry['pred_probs'] = np.asarray(dct['pred_probs'], dtype=float)
            pred_entry['image_path'] = dct['image_path']
            pred_list.append(pred_entry)


    if params['qual'] is not None:
        if not osp.exists(params['qual']):
            print('{} does not exist. Creating it ...'.format(params['qual']))
            os.mkdir(params['qual'])
        i = 0
        for pred in pred_list:
            sys.stdout.write("Processing %d/%d (%.2f%% done) \r" % (i + 1, len(pred_list),
                                                                    (i + 1) * 100.0 / len(pred_list)))
            sys.stdout.flush()
            #image_path = osp.join(DS_ROOT, pred['image_path'])
            #im = Image.open(image_path)

            #fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(20, 15))

            #ax = ax1
            #ax.imshow(im)
            #ax.axis('off')

            #ax = ax2
            #text_str = ''
            pred_probs = pred['pred_probs']
            top_10_inds = np.argsort(-pred_probs)[:10]
            for aidx in top_10_inds:
                if pred_probs[aidx] > 0.1:
                    #text_str += '{:<30} {:.3f}\n'.format(idx_to_attr_id[aidx], pred_probs[aidx])
                    attr_id_count[idx_to_attr_id[aidx]] += 1
                    attr_id_conf[idx_to_attr_id[aidx]].append(pred_probs[aidx])
            #ax.set_xlim(xmin=0, xmax=1)
            #ax.set_ylim(ymin=0, ymax=1)
            #ax.text(0.0, 0.5, text_str, fontsize='xx-large')
            #ax.axis('off')

            #plt.tight_layout()

            #_, im_name = osp.split(image_path)
            #out_path = osp.join(params['qual'], im_name)
            #plt.savefig(out_path, bbox_inches='tight')
            #plt.close()
            i += 1
    print attr_id_count
    attr_id_conf = {k: v for k, v in attr_id_conf.items()}
    #attr_id_conf_var = {k: np.var(np.asarray(v,dtype=np.float)) for k, v in attr_id_conf.items()}

    save_obj(attr_id_count, 'attr_id_count_top10_0.1')
    save_obj(attr_id_conf, 'attr_id_conf_top10_0.1')
    #save_obj(attr_id_conf_var, 'attr_id_conf_var_top10_0.1')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-deploy", type=str, default='models/resnet-pap/deploy.prototxt', help="Path to caffe deploy proto file")
    parser.add_argument("-weights", type=str, default='models/resnet-pap/resnet_pap.caffemodel', help="Path to .caffemodel")
    parser.add_argument("-outfile", type=str, default='pap_out.txt', help="Prefix for path to write the classification output to")
    parser.add_argument("-pred_file", type=str, default='pap_out.txt', help="Prefix for path to write the classification output to")

    parser.add_argument("-d", "--device", type=int, choices=[0, 1, 2, 3, 4, 5, 6, 7], default=1, help="GPU device id")
    parser.add_argument("-b", "--batch_size", type=int, choices=range(512), default=32, help="Batch size")
    parser.add_argument("-q", "--qual", type=str, default='qual', help="Path to write qualitative results")

    args = parser.parse_args()

    params = vars(args)

    # Initialize Network -----------------------------------------------------------------------------------------------
    caffe.set_device(params['device'])
    caffe.set_mode_gpu()

    model_def = params['deploy']
    model_weights = params['weights']

    net = caffe.Net(model_def,  # defines the structure of the model
                    model_weights,  # contains the trained weights
                    caffe.TEST)  # use test mode (e.g., don't perform dropout)

    # Set up transformer -----------------------------------------------------------------------------------------------
    # load the mean ImageNet image (as distributed with Caffe) for subtraction
    mu = np.load(osp.join(CAFFE_ROOT, 'python/caffe/imagenet/ilsvrc_2012_mean.npy'))
    mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
    # mu = np.asarray([111.0, 102.0, 116.0])
    # mu = np.asarray([104.0, 117.0, 123.0])
    # print 'mean-subtracted values:', zip('BGR', mu)

    # create transformer for the input called 'data'
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

    transformer.set_transpose('data', (2, 0, 1))  # move image channels to outermost dimension
    transformer.set_mean('data', mu)  # subtract the dataset-mean value in each channel
    transformer.set_raw_scale('data', 255)  # rescale from [0, 1] to [0, 255]
    transformer.set_channel_swap('data', (2, 1, 0))  # swap channels from RGB to BGR

    filenames = []
    #for path, subdirs, files in os.walk('/home/wuzhenyu_sjtu/DAN/SBU/SBU_frames'):
    #for path, subdirs, files in os.walk('/home/wuzhenyu_sjtu/vpa/hmdb_pap/hmdb51_frames'):
    for path, subdirs, files in os.walk('/home/wuzhenyu_sjtu/vpa/pap_real_data/UCF-101_frames'):
        for name in files:
            filenames.append(os.path.join(path, name))

    print(filenames)
    # Classify ---------------------------------------------------------------------------------------------------------
    classify_paths(net, transformer, filenames, params['outfile'], batch_size=params['batch_size'])
    evaluate(params)

if __name__ == '__main__':
    main()
