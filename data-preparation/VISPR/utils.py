import csv
import os.path as osp
import numpy as np

def load_attributes(attributes_path):
    """
    Returns mappings: {attribute_id -> attribute_name} and {attribute_id -> idx}
    where attribute_id = 'aXX_YY' (string),
    attribute_name = description (string),
    idx \in [0, 67] (int)
    :return:
    """
    attr_id_to_name = dict()
    attr_id_to_idx = dict()

    with open(attributes_path, 'r') as fin:
        ts = csv.DictReader(fin, delimiter='\t')
        rows = filter(lambda r: r['idx'] is not '', [row for row in ts])

        for row in rows:
            attr_id_to_name[row['attribute_id']] = row['description']
            attr_id_to_idx[row['attribute_id']] = int(row['idx'])

    return attr_id_to_name, attr_id_to_idx


def labels_to_vec(labels, attr_id_to_idx):
    n_labels = len(set(attr_id_to_idx.values()))
    print(n_labels)
    label_vec = np.zeros(n_labels)
    count = 0
    for attr_id in labels:
        if attr_id in attr_id_to_idx:
            label_vec[attr_id_to_idx[attr_id]] = 1
            count += 1
    if count == 0:
        label_vec[0] = 1
    return label_vec
