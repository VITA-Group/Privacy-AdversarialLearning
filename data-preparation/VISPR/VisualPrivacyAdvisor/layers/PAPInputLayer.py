# imports
import json
import time
import pickle
import scipy.misc
import skimage.io
import os
import sys
import csv
import copy
from collections import Counter

import numpy as np
import os.path as osp

import json
from random import shuffle
from threading import Thread
from PIL import Image
from scipy.misc import imread

CAFFE_ROOT = '/BS/orekondy/work/opt/caffe-wloss/'
sys.path.insert(0, os.path.join(CAFFE_ROOT, 'python'))
import caffe

sys.path.insert(1, CAFFE_ROOT + 'examples/pycaffe/layers')   # the datalayers we will use are in this directory.
sys.path.insert(1, CAFFE_ROOT + 'examples/pycaffe')   # the tools file is in this folder
from tools import SimpleTransformer

ATTRIBUTE_PATH = '/home/orekondy/work/blur_personal/multi_label_anno_tool/static/attributes_v2.tsv'
SURVEY_RES_PATH = '/home/orekondy/work2/blur_personal/survey_graphs/survey_v5_res50.tsv'
NUM_LABELS = 104

SAFE_ATTR_ID = 'a0_safe'
SAFE_WEIGHT = 0.0


def load_attributes(attribute_path=ATTRIBUTE_PATH):
    attr_id_to_categ_id = dict()
    attr_id_to_name = dict()
    attr_id_to_mode = dict()
    categ_id_to_name = dict()
    attr_id_to_weight = dict()

    with open(attribute_path, 'r') as fin:
        ts = csv.DictReader(fin, delimiter='\t')
        rows = filter(lambda r: r['category_id'] is not '', [row for row in ts])

        for row in rows:
            attr_id_to_categ_id[row['attribute_id']] = row['category_id']
            attr_id_to_name[row['attribute_id']] = row['attribute_name']
            attr_id_to_mode[row['attribute_id']] = row['mode']
            categ_id_to_name[row['category_id']] = row['category_name']
            attr_id_to_weight[row['attribute_id']] = row['weight']

    if os.path.exists(SURVEY_RES_PATH):
        attr_id_to_weight = survey_to_weights(SURVEY_RES_PATH, use_attr=set(attr_id_to_name.keys()))

    # Handle 'safe' label
    attr_id_to_weight[SAFE_ATTR_ID] = SAFE_WEIGHT
    attr_id_to_mode[SAFE_ATTR_ID] = 'visual'
    attr_id_to_name[SAFE_ATTR_ID] = 'safe'

    return attr_id_to_categ_id, attr_id_to_name, attr_id_to_mode, categ_id_to_name, attr_id_to_weight


def survey_to_weights(survey_res_path=SURVEY_RES_PATH, use_attr=None):
    attr_id_to_weight = dict()

    with open(survey_res_path, 'r') as fin:
        ts = csv.DictReader(fin, delimiter='\t')
        rows = [row for row in ts]

        for row in rows:
            attr_id = row['attribute_id']

            if use_attr is not None and attr_id not in use_attr:
                # There might be some control attribute_ids. So, skip these.
                continue

            survey_res = [float(row[v]) for v in ('1', '2', '3', '4', '5')]
            weighted_avg = sum([v1*v2 for (v1, v2) in zip([1, 2, 3, 4, 5], survey_res)]) / 30.0

            attr_id_to_weight[attr_id] = weighted_avg

    return attr_id_to_weight


def get_filename(filepath, drop_ext=False):
    _, filename_with_ext = os.path.split(filepath)
    if not drop_ext:
        return filename_with_ext
    else:
        filename_without_ext, _ = os.path.splitext(filename_with_ext)
        return filename_without_ext


def load_attr_to_idx():
    attr_id_to_categ_id, attr_id_to_name, attr_id_to_mode, categ_id_to_name, attr_id_to_weight = load_attributes()
    # Attributes are stored as "aNN_name". Sort using NN
    attr_list = sorted(attr_id_to_name.keys(), key=lambda x: int(x.split('_')[0][1:]))
    # attr_list = ['safe', ] + attr_list   # Treat 'safe' as another class

    attr_id_to_idx = dict()
    for idx, attr_id in enumerate(attr_list):
        attr_id_to_idx[attr_id] = idx

    return attr_id_to_idx


def attribute_set_to_vec(attr_id_to_idx, attr_set, is_safe):
    num_labels = len(attr_id_to_idx)
    label_vec = np.zeros(num_labels, dtype=np.float32)
    for attr_id in attr_set:
        if attr_id in attr_id_to_idx:
            idx = attr_id_to_idx[attr_id]
            label_vec[idx] = 1
    if len(attr_set) == 0:
        assert is_safe
        idx = attr_id_to_idx[SAFE_ATTR_ID]
        label_vec[idx] = 1
    return label_vec


def get_w2idx(dictlist, attr_id_to_weight):
    """
    1. Create a mapping of weight (int) -> attribute_idx
       weight of example = max(weight of attribute i in example)
    2. When sampling next image:
      a. Sample weight ~ [1, 2, 3, 4, 5]
      b. Sample an example corresponding to this weight
    Maintain a dict:
        {
            1: [3, 10, 4, ...],
            2: [45, 11, 90, ...],
            ...
        }
    and pop an idx from the list when asked for next image
    """
    weight_to_idx_list = {}

    for idx, this_anno in enumerate(dictlist):
        if 'labels' in this_anno:
            this_attr_list = this_anno['labels']
        else:
            this_attr_list = []
            for categ_id, attr_id_list in this_anno['attributes'].iteritems():
                this_attr_list += attr_id_list
        attr_id_set = set(this_attr_list)
        # What's the weight of this training example?
        if len(attr_id_set) == 0:
            idx_weight = 1
        else:
            idx_weight = max([attr_id_to_weight.get(x, 1) for x in attr_id_set])
        # This is a float. Cast it to int by rounding off.
        idx_weight = int(np.round(idx_weight))
        if idx_weight not in weight_to_idx_list:
            weight_to_idx_list[idx_weight] = [idx, ]
        else:
            weight_to_idx_list[idx_weight].append(idx)

    for w in sorted(weight_to_idx_list.keys()):
        shuffle(weight_to_idx_list[w])
        print '{} -> {}'.format(w, len(weight_to_idx_list[w]))

    return weight_to_idx_list


def get_class2idx(dictlist):
    """
    1. Create a mapping of LABEL (attr_id) -> DICT_IDX
    2. When sampling next image:
      a. Sample class ~ [attr_1, attr_2, ..., attr_L]
      b. Sample an example corresponding to this label
    Maintain a dict:
        {
            attr_1: [3, 10, 4, ...],
            attr_2: [45, 11, 90, ...],
            ...
        }
    and pop an idx from the list when asked for next image
    """
    class_to_idx_list = {}

    for idx, this_anno in enumerate(dictlist):
        if 'labels' in this_anno:
            this_attr_list = this_anno['labels']
        else:
            this_attr_list = []
            for categ_id, attr_id_list in this_anno['attributes'].iteritems():
                this_attr_list += attr_id_list
        attr_id_set = set(this_attr_list)

        for attr_id in attr_id_set:
            if attr_id not in class_to_idx_list:
                class_to_idx_list[attr_id] = [idx, ]
            else:
                class_to_idx_list[attr_id].append(idx)

    for attr_id in sorted(class_to_idx_list.keys(), key=lambda x: x.split('_')[0][1:]):
        shuffle(class_to_idx_list[attr_id])
        print '{} -> {}'.format(attr_id, len(class_to_idx_list[attr_id]))

    return class_to_idx_list


class PAPMultilabelDataLayerSync(caffe.Layer):

    """
    This is a simple synchronous datalayer for training a multilabel model on
    PASCAL.
    """

    def setup(self, bottom, top):

        self.top_names = ['data', 'label']

        # === Read input parameters ===

        # params is a python dictionary with layer parameters.
        # Example param_str: param_str:
        # "{\'anno_list\': \'/BS/orekondy2/work/blur_personal/experiments/8k/train_6k.txt\', \'im_shape\': [227, 227], \'batch_size\': 128}"
        params = eval(self.param_str)

        # Check the parameters for validity.
        check_params(params)

        # store input as class variables
        self.batch_size = params['batch_size']

        self.num_labels = params.get('nlabels', NUM_LABELS)

        # Create a batch loader to load the images.
        self.batch_loader = BatchLoader(params, None)

        # Weighted loss
        self.wloss = bool(params.get('wloss', 0))

        # Return user preferences
        self.user_prefs = params.get('user_prefs', None)
        self.n_users = 0

        self.pool = params.get('pool', 'max')
        assert self.pool in ['sum', 'max', 'avg']

        # === reshape tops ===
        # since we use a fixed input image size, we can shape the data layer
        # once. Else, we'd have to do it in the reshape call.
        top[0].reshape(
            self.batch_size, 3, params['im_shape'][0], params['im_shape'][1])
        # Note the 20 channels (because PASCAL has 20 classes.)
        top[1].reshape(self.batch_size, 68)

        if self.wloss or len(top) > 2:
            top[2].reshape(68)

        if self.user_prefs is not None:
            self.user_pref_mat = self.batch_loader.get_user_prefs()
            n_attr, self.n_users = self.user_pref_mat.shape
            print 'self.n_users = ', self.n_users
            top[3].reshape(self.batch_size, self.n_users)

        print_info("PAPMultilabelDataLayerSync", params)

    def forward(self, bottom, top):
        """
        Load data.
        """

        # print '******** top[0].shape = ', top[0].shape
        # print '******** top[1].shape = ', top[1].shape

        # Create a matrix N x A: Labels for each image
        # This *may* be needed later on depending on whether user preferences are specified
        n_attr = len(self.batch_loader.get_attr_id_list())
        img_attr_mat = np.zeros((self.batch_size, n_attr))

        for itt in range(self.batch_size):
            # Use the batch loader to load the next image.
            im, multilabel = self.batch_loader.load_next_image()

            assert multilabel.shape[0] == n_attr
            img_attr_mat[itt] = multilabel[:]

            # Add directly to the caffe data layer
            top[0].data[itt, ...] = im
            top[1].data[itt, ...] = multilabel

        if self.wloss:
            top[2].data[...] = self.batch_loader.get_weights()

        if self.user_prefs is not None:
            '''
            top[3] needs to be a matrix, say X, of dims batch_size x n_users
            s.t. X[i, j] = Privacy score of image i for user j
            '''
            if self.pool in ['sum', 'avg']:
                user_scores_mat = np.dot(img_attr_mat, self.user_pref_mat)
                if self.pool == 'avg':
                    # Get a N-dim vector representing #attributes in each example
                    y_card_vec = np.sum(img_attr_mat, axis=1)
                    user_scores_mat /= y_card_vec[:, None]
            else:
                user_scores_mat = np.zeros((self.batch_size, self.n_users))
                for n in range(self.batch_size):
                    for u in range(self.n_users):
                        user_scores_mat[n, u] = np.max(img_attr_mat[n, :] * self.user_pref_mat[:, u])
            top[3].data[...] = user_scores_mat

    def reshape(self, bottom, top):
        """
        There is no need to reshape the data, since the input is of fixed size
        (rows and columns)
        """
        pass

    def backward(self, top, propagate_down, bottom):
        """
        These layers does not back propagate
        """
        pass


class BatchLoader(object):

    """
    This class abstracts away the loading of images.
    Images can either be loaded singly, or in a batch. The latter is used for
    the asyncronous data layer to preload batches while other processing is
    performed.
    """

    def __init__(self, params, result):
        self.result = result
        self.batch_size = params['batch_size']
        self.anno_list = params['anno_list']
        self.im_shape = params['im_shape']
        self.label_shortlist_path = params['label_shortlist']
        self.num_labels = params.get('nlabels', NUM_LABELS)
        self.memimages = bool(params.get('memimages', 0))
        self.img_transform = params.get('img_transform', 'resize')
        self.ynorm = bool(params.get('ynorm', 0))   # y := y / ||y||_1
        self.wloss = bool(params.get('wloss', 0))
        self.user_prefs = params.get('user_prefs', None)
        self.scale_user_pref = bool(params.get('scale_user_pref', 0))

        # Possible options:
        # 'uniform' (Default) : Sample uniformly
        # 'weighted': Sample a weight uniformly (usually between 1-5). Then sample an example from on of these.
        self.sampling = params.get('sampling', 'uniform')

        if self.label_shortlist_path is not None:
            self.attr_id_to_idx = dict()
            self.attr_id_to_weight = dict()
            with open(self.label_shortlist_path, 'r') as f:
                f.readline()   # Skip header line
                for line in f:
                    idx, attr_id, count, weight = line.strip().split('\t')
                    idx = int(idx)
                    count = int(count)
                    weight = float(weight)
                    self.attr_id_to_idx[attr_id] = idx
                    self.attr_id_to_weight[attr_id] = weight
        else:
            assert False, "Not Supported"
            # self.attr_id_to_idx = load_attr_to_idx()

        self.attr_id_list = self.attr_id_to_idx.keys()
        self.n_attr = len(self.attr_id_list)

        self.user_mat = None
        if self.user_prefs is not None:
            '''
            This is a file of the format:
            <attribute_id>  <attribute_name>    <score_1>   <score_2> .... <score_U>
            where U = # of users
            score_U indicates how important this attribute is to him/her
            '''
            with open(self.user_prefs) as uf:
                uf.readline()   # Skip header line
                pref_dct = dict()   # Store mapping: attr_id = [..., score_i, ...]
                for line in uf:
                    if line.strip() == '':
                        continue
                    tokens = line.strip().split('\t')
                    attr_id = tokens[0]
                    attr_name = tokens[1]
                    scores = [float(s) for s in tokens[2:]]
                    if attr_id in self.attr_id_to_idx:
                        pref_dct[attr_id] = scores

                # Check n_users is consistent
                n_users = len(pref_dct.values()[0])
                assert all([n_users == len(x) for x in pref_dct.values()]), Counter([len(x) for x in pref_dct.values()])

                # Manually fill-in safe
                pref_dct[SAFE_ATTR_ID] = np.ones(n_users) * SAFE_WEIGHT
                # Make sure we have preferences for all attributes that we need
                assert all([pref_attr_id in self.attr_id_to_idx for pref_attr_id in pref_dct.keys()])

                # Represent as a matrix: A x U
                # Where col_j represents attribute preferences for user j
                n_attr = len(self.attr_id_to_idx)
                self.user_mat = np.zeros((n_attr, n_users))
                for attr_id, idx in self.attr_id_to_idx.iteritems():
                    attr_scores = pref_dct[attr_id]
                    self.user_mat[idx] = attr_scores

            print 'User preferences: '
            print self.user_mat
            print 'user_mat.shape = ', self.user_mat.shape

            # Normalize user_mat
            if self.scale_user_pref:
                self.user_mat -= 2.5   # Assuming mean of scores = 2.5, so scale to [-2.5, 2.5]
                self.user_mat /= 2.5   # Scale to [-1, 1]

        # Store the list of annotation files as indexlist
        self.indexlist = [line.rstrip('\n') for line in open(self.anno_list)]

        if self.memimages:
            print "Loading images into memory"
        print "Loading {} annotations".format(len(self.indexlist))

        # Store each image-label object as a dict
        # But, do not store the images. Only store the image file path
        self.dictlist = [json.load(open(aidx)) for aidx in self.indexlist]
        shuffle(self.dictlist)

        # Create a weight vector
        self.idx_to_attr_id = {v: k for k, v in self.attr_id_to_idx.iteritems()}
        self.idx_to_weight = np.ones(68)
        if self.wloss:
            for idx in sorted(self.idx_to_attr_id.keys()):
                attr_id = self.idx_to_attr_id[idx]
                self.idx_to_weight[idx] = self.attr_id_to_weight[attr_id]

        print 'Class weights: '
        print self.idx_to_weight

        if self.sampling == 'weighted':
            '''
            1. Create a mapping of WEIGHT (int) -> attribute_idx
               weight of example = max(weight of attribute i in example)
            2. When sampling next image:
              a. Sample weight ~ [1, 2, 3, 4, 5]
              b. Sample an example corresponding to this weight
            Maintain a dict:
                {
                    1: [3, 10, 4, ...],
                    2: [45, 11, 90, ...],
                    ...
                }
            and pop an idx from the list when asked for next image
            '''
            self.weight_to_idx_list = get_w2idx(self.dictlist, self.attr_id_to_weight)
            # Maintain a copy of this, because it will mutate in each iteration (pop() to consume)
            self.org_weight_to_idx_list = copy.deepcopy(self.weight_to_idx_list)
        elif self.sampling == 'class_weighted':
            '''
            1. Create a mapping of LABEL (attr_id) -> DICT_IDX
            2. When sampling next image:
              a. Sample class ~ [attr_1, attr_2, ..., attr_L]
              b. Sample an example corresponding to this label
            Maintain a dict:
                {
                    attr_1: [3, 10, 4, ...],
                    attr_2: [45, 11, 90, ...],
                    ...
                }
            and pop an idx from the list when asked for next image
            '''
            self.class_to_idx_list = get_class2idx(self.dictlist)
            # Maintain a copy of this, because it will mutate in each iteration (pop() to consume)
            self.org_class_to_idx_list = copy.deepcopy(self.class_to_idx_list)
        else:
            self._cur = 0  # current image
            self.weight_to_idx_list = None
            self.class_to_idx_list = None

        # Add to each dict the label vector
        for idx, this_anno in enumerate(self.dictlist):
            # Prepare the multilabel
            # Get the list of attributes this corresponds to
            if 'labels' in this_anno:
                attr_set = set(this_anno['labels'])
            else:
                this_attr_list = []
                for categ_id, attr_id_list in this_anno['attributes'].iteritems():
                    this_attr_list += attr_id_list
                attr_set = set(this_attr_list)
            multilabel = attribute_set_to_vec(self.attr_id_to_idx, attr_set, is_safe=this_anno['safe'])
            if self.ynorm and np.sum(multilabel) > 0:
                multilabel /= np.sum(multilabel)
            assert np.sum(multilabel) > 0, 'Failed: np.sum(multilabel) > 0'
            this_anno['label_vec'] = multilabel

            image_path = this_anno['image_path']
            image_resized_path = image_path.replace('/images_chunks/', '/images_chunks_resized/')
            if os.path.exists(image_resized_path):
                this_anno['image_path'] = image_resized_path

            if self.memimages:
                im = imread(this_anno['image_path'])
                if len(im.shape) == 2:
                    # This is a grayscale image
                    im = np.asarray(Image.open(this_anno['image_path']).convert('RGB'))
                elif len(im.shape) == 3 and im.shape[2] == 4:
                    # CMYK Image
                    im = np.asarray(Image.open(this_anno['image_path']).convert('RGB'))

                if self.img_transform == 'resize':
                    # Resize the image to the required shape
                    im = scipy.misc.imresize(im, self.im_shape)

                this_anno['im'] = im

                if idx % 100 == 0:
                    sys.stdout.write("processing %d/%d (%.2f%% done)   \r" % (idx, len(self.dictlist), idx * 100.0 / len(self.dictlist)))
                    sys.stdout.flush()

        print 'multilabel.shape = ', multilabel.shape
        # this class does some simple data-manipulations
        self.transformer = SimpleTransformer(mean=[104, 117, 123])

        print "BatchLoader initialized with {} images".format(len(self.indexlist))

    def get_weights(self):
        return self.idx_to_weight

    def get_user_prefs(self):
        return self.user_mat.copy()

    def get_attr_id_list(self):
        return self.attr_id_list

    def load_next_image(self):
        """
        Load the next image in a batch.
        """

        # Sample image -------------------------------------------------------------------------------------------------
        # Choose which idx in dctlist to read
        # The next block should fill this in
        if self.sampling == 'weighted':
            # 1. Sample a weight
            this_w = np.random.choice(self.weight_to_idx_list.keys())
            # 2.a. Is an image available for this weight. If not,
            if len(self.weight_to_idx_list[this_w]) == 0:
                # Copy from the original mapping
                self.weight_to_idx_list = copy.deepcopy(self.org_weight_to_idx_list)
                # Shuffle indices
                for w in sorted(self.weight_to_idx_list.keys()):
                    shuffle(self.weight_to_idx_list[w])
            # 2.b. Get the next index
            next_idx = self.weight_to_idx_list[this_w].pop()
        elif self.sampling == 'class_weighted':
            # 1. Sample a label
            this_attr_id = np.random.choice(self.class_to_idx_list.keys())
            # 2a. Is there a training example available for this weight? If not,
            if len(self.class_to_idx_list[this_attr_id]) == 0:
                # Copy from original mapping
                self.class_to_idx_list = copy.deepcopy(self.org_class_to_idx_list)
                # Shuffle them
                for ai in self.class_to_idx_list:
                    shuffle(self.class_to_idx_list[ai])
            # 2b. Get next index
            next_idx = self.class_to_idx_list[this_attr_id].pop()
        else:
            # Did we finish an epoch?
            if self._cur == len(self.dictlist):
                self._cur = 0
                next_idx = self._cur
                shuffle(self.dictlist)
            else:
                next_idx = self._cur
                self._cur += 1

        dct = self.dictlist[next_idx]  # Get the anno

        # Load image ---------------------------------------------------------------------------------------------------
        image_path = dct['image_path']

        multilabel = dct['label_vec']
        assert multilabel.shape[0] == self.num_labels, 'multilabel.shape[0] ({}) != self.num_labels ({})'.format(multilabel.shape[0], self.num_labels)

        # Load an image
        if 'im' in dct:
            im = dct['im']
        else:
            im = imread(image_path)
            if len(im.shape) == 2:
                # This is a grayscale image
                im = np.asarray(Image.open(image_path).convert('RGB'))
            elif len(im.shape) == 3 and im.shape[2] == 4:
                # CMYK Image
                im = np.asarray(Image.open(image_path).convert('RGB'))
        org_shape = im.shape

        # Resize/Transform image ---------------------------------------------------------------------------------------
        if self.img_transform == 'resize':
            # Resize the image to the required shape
            im = scipy.misc.imresize(im, self.im_shape)
        elif self.img_transform == 'rand_crop':
            # Take a random crop of size self.im_shape
            # im.shape = [H, W, 3]
            img_h, img_w, _ = im.shape
            crop_h, crop_w = self.im_shape

            # print 'Processing file: ', image_path
            # print 'Old (w, h): ', (img_w, img_h)

            if img_w < crop_w:
                new_w = crop_w
                new_h = int(np.round(img_h * (new_w / float(img_w))))   # Scale height to same aspect ratio
                im = scipy.misc.imresize(im, (new_h, new_w))
                img_w, img_h = new_w, new_h
                # print 'New (w, h): ', (img_w, img_h)

            if img_h < crop_h:
                new_h = crop_h
                new_w = int(np.round(img_w * (new_h / float(img_h))))
                im = scipy.misc.imresize(im, (new_h, new_w))
                img_w, img_h = new_w, new_h
                # print 'New (w, h): ', (img_w, img_h)

            # Sample (x1, y1) i.e, top-left point of the image
            x1 = np.random.randint(low=0, high=(img_h - crop_h - 1))
            y1 = np.random.randint(low=0, high=(img_w - crop_w - 1))
            # Crop a window given this point
            x2 = x1 + crop_h
            y2 = y1 + crop_w
            im = im[x1:x2, y1:y2, :]

            # print '(x1, y1) = ', (x1, x2)
            # print 'Cropped (w, h): ', (x2-x1, y2-y1)
            # print 'im.shape = ', im.shape

        # do a simple horizontal flip as data augmentation
        flip = np.random.choice(2)*2-1
        im = im[:, ::flip, :]

        transformed_im = self.transformer.preprocess(im)

        return transformed_im, multilabel


def check_params(params):
    """
    A utility function to check the parameters for the data layers.
    """
    # assert 'split' in params.keys(
    # ), 'Params must include split (train, val, or test).'

    required = ['batch_size', 'anno_list', 'im_shape']
    for r in required:
        assert r in params.keys(), 'Params must include {}'.format(r)


def print_info(name, params):
    """
    Output some info regarding the class
    """
    print "{} initialized for split: {}, with bs: {}, im_shape: {}.".format(
        name,
        params['anno_list'],
        params['batch_size'],
        params['im_shape'])
