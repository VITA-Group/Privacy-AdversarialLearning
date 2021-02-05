#!/usr/bin/env python
# coding: utf-8

# In[24]:


'''
All labeled videos in PA-HMDB51 are used as testing set.
The rest videos in HMDB51 are used as training set.
This script gets the file names of PA_HMDB51 dataset.
'''

import numpy as np 
import matplotlib.pyplot as plt
plt.style.use(['/Users/wuzhenyu/.matplotlib/stylelib/science.mplstyle', 
               '/Users/wuzhenyu/.matplotlib/stylelib/ieee.mplstyle'])
import json, os, csv

from calculate_statistics import attribute_list, attribute_value_dict

attr_list = [attr.replace('_', ' ') for attr in attribute_list]
attr_value_dict = {k.replace('_', ' '):v for k,v in attribute_value_dict.items()}
attr_act_corl_mat = json.load(open(os.path.join("attr_act_corl_mat.json")))
attr_act_corl_mat = {k.replace('_', ' '):v for k,v in attr_act_corl_mat.items()}
attr_value_num = json.load(open(os.path.join("attr_value_num.json")))
attr_value_num = {k.replace('_', ' '):v for k,v in attr_value_num.items()}

r = csv.reader(open('act_vid_dist.csv', 'r'), delimiter=',')
print('r', type(r))
for i, row in enumerate(r):
    if i == 0:
        action_list = row
    elif i == 1:
        action_number = np.array(row).astype(int)
    print(row)
print(action_list)

fontsize_ticks = 48
fontsize_legend = 48
fontsize_axis_labels = 48


# In[25]:


'''
Plot action - Number of Frames (i.e., Fig 8 in camera ready version)
'''
def plot_action_distribution():

    plt.figure(figsize=(18, 24))


    ax = plt.subplot(1,1,1)
    plt.bar(action_list, action_number)
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    plt.xticks(fontsize=fontsize_ticks, weight='medium')
    plt.yticks(fontsize=fontsize_ticks, weight='medium')
    font = {'family': 'Times New Roman',
            'color':  'black',
            'weight': 'semibold',
            'size': fontsize_axis_labels,
    }
    plt.xlabel('Actions', fontdict=font)
    plt.ylabel('Number of Videos', fontdict=font)
    # plt.tight_layout()
    plt.gcf().subplots_adjust(bottom=0.18, top=0.98, left=0.05, right=0.95)
    plt.savefig('action_distribution.png')


def plot_action_distribution_transposed():    

    plt.figure(figsize=(24, 48))
    
    ax = plt.subplot(1,1,1)

    
    y_pos = np.arange(len(action_list))
    
    ax.barh(y_pos, action_number, align='center', color='b')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(action_list)
    ax.invert_yaxis()
    
    #plt.tick_params(labelsize=fontsize_ticks)
    plt.xticks(fontsize=fontsize_ticks, weight='medium')
    plt.yticks(fontsize=fontsize_ticks, weight='medium')
    ax.tick_params(labelsize=fontsize_ticks)
    
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.tick_params(axis='y', which='both', left=False)
    ax.tick_params(axis='x', which='both', bottom=True, width=4, colors='k')
    font = {'family': 'Times New Roman',
            'color':  'black',
            'weight': 'semibold',
            'size': fontsize_axis_labels,
    }
    ax.set_ylabel('Actions', fontdict=font)
    ax.set_xlabel('Number of Videos', fontdict=font)
    plt.autoscale(tight=True)
    plt.tight_layout()
    #plt.gcf().subplots_adjust(bottom=0.15, top=0.85, left=0.15, right=0.85)
    plt.savefig('action_distribution.pdf',bbox_inches="tight", pad_inches=0)
    
        

if __name__ == '__main__':
    # plot_action_distribution()
    plot_action_distribution_transposed()

