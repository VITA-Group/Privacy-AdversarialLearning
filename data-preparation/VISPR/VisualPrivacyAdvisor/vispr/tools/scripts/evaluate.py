#!/usr/bin/python
"""This is a short description.

Replace this with a more detailed description of what this file contains.
"""
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
from sklearn.metrics import average_precision_score

from vispr import DS_ROOT
from vispr.tools.common.utils import *

__author__ = "Tribhuvanesh Orekondy"
__maintainer__ = "Tribhuvanesh Orekondy"
__email__ = "orekondy@mpi-inf.mpg.de"
__status__ = "Development"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pred_file", type=str, help="File path to list of predictions")
    parser.add_argument("-c", "--class_scores", type=str, default=None, help="Path to write class-specific APs")
    parser.add_argument("-q", "--qual", type=str, default=None, help="Path to write qualitative results")
    args = parser.parse_args()

    params = vars(args)

    # Load Attributes --------------------------------------------------------------------------------------------------
    attr_id_to_name, attr_id_to_idx = load_attributes()
    idx_to_attr_id = {v: k for k, v in attr_id_to_idx.iteritems()}
    n_attr = len(attr_id_to_idx)

    # Load predictions -------------------------------------------------------------------------------------------------
    # Construct a list of dicts containing: GT labels, Prediction probabilities, Image path
    pred_list = []
    with open(params['pred_file'], 'r') as f:
        for _line in f:
            line = _line.strip()
            dct = json.loads(line)

            pred_entry = dict()
            pred_entry['pred_probs'] = np.asarray(dct['pred_probs'], dtype=float)

            # Read image_path and gt_labels from annotation
            anno_path = dct['anno_path'] if osp.exists(dct['anno_path']) else osp.join(DS_ROOT, dct['anno_path'])
            with open(anno_path) as jf:
                anno = json.load(jf)

                # Get the list of attributes this corresponds to
                attr_set = set(anno['labels'])
                attr_vec = labels_to_vec(attr_set, attr_id_to_idx)

                pred_entry['image_path'] = anno['image_path']
                pred_entry['gt_labels'] = attr_vec
                pred_entry['anno_path'] = dct['anno_path']

            pred_list.append(pred_entry)

    # Convert to matrix ------------------------------------------------------------------------------------------------
    # Create a NxM matrix. Each row represents the class-probabilities for the M classes.
    # In case of GT, they are 1-hot encoded
    gt_mat = np.array([d['gt_labels'] for d in pred_list])
    pred_probs_mat = np.array([d['pred_probs'] for d in pred_list])

    # Drop examples where gt contains no relevant attributes (when testing on a partial set)
    # non_empty_gt_idx = np.where(np.sum(gt_mat, axis=1) > 0)[0]
    # pred_probs_mat = pred_probs_mat[non_empty_gt_idx, :]
    # gt_mat = gt_mat[non_empty_gt_idx, :]

    # Evaluate Overall Attribute Prediction ----------------------------------------------------------------------------
    n_examples, n_labels = gt_mat.shape
    print '# Examples = ', n_examples
    print '# Labels = ', n_labels
    print 'Macro MAP = {:.2f}'.format(100 * average_precision_score(gt_mat, pred_probs_mat, average='macro'))

    if params['class_scores'] is not None:
        cmap_stats = average_precision_score(gt_mat, pred_probs_mat, average=None)
        with open(params['class_scores'], 'w') as wf:
            wf.write('\t'.join(['attribute_id', 'attribute_name', 'num_occurrences', 'ap']) + '\n')
            for idx in range(n_labels):
                attr_id = idx_to_attr_id[idx]
                attr_name = attr_id_to_name[attr_id]
                attr_occurrences = np.sum(gt_mat, axis=0)[idx]
                ap = cmap_stats[idx]

                wf.write('{}\t{}\t{}\t{}\n'.format(attr_id, attr_name, attr_occurrences, ap*100.0))

    if params['qual'] is not None:
        if not osp.exists(params['qual']):
            print '{} does not exist. Creating it ...'.format(params['qual'])
            os.mkdir(params['qual'])
        for pred in pred_list:
            image_path = pred['image_path']
            im = Image.open(image_path)

            fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(20, 15))

            ax = ax1
            ax.imshow(im)
            ax.axis('off')

            ax = ax2
            text_str = ''
            pred_probs = pred['pred_probs']
            top_10_inds = np.argsort(-pred_probs)[:10]
            for aidx in top_10_inds:
                text_str += '{:<30} {:.3f}\n'.format(idx_to_attr_id[aidx], pred_probs[aidx])
            ax.set_xlim(xmin=0, xmax=1)
            ax.set_ylim(ymin=0, ymax=1)
            ax.text(0.0, 0.5, text_str, fontsize='xx-large')
            ax.axis('off')

            plt.tight_layout()

            _, im_name = osp.split(image_path)
            out_path = osp.join(params['qual'], im_name)
            plt.savefig(out_path, bbox_inches='tight')
            plt.close()


if __name__ == '__main__':
    main()