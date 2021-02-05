#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''
All labeled videos in PA-HMDB51 are used as testing set.
The rest videos in HMDB51 are used as training set.
This script gets the file names of PA_HMDB51 dataset.
'''

import numpy as np 
import matplotlib.pyplot as plt
import matplotlib
plt.style.use(['/Users/wuzhenyu/.matplotlib/stylelib/science.mplstyle', 
               '/Users/wuzhenyu/.matplotlib/stylelib/ieee.mplstyle'])
import json, os, csv

from calculate_statistics import attribute_list

attribute_value_dict = {
    'skin_color': ['0', '1', '2', '3', '4'],
    'relationship': ['0', '1'],
    'face': ['0', '1', '2'], 
    'nudity': ['0', '1' ,'2'],
    'gender': ['0', '1', '2', '3']
    }

print(attribute_value_dict)

r = csv.reader(open('act_vid_dist.csv', 'r'), delimiter=',')
print('r', type(r))
for i, row in enumerate(r):
    if i == 0:
        action_list = row
    elif i == 1:
        action_number = np.array(row).astype(int)
    print(row)
print(action_list)

attr_list = [attr.replace('_', ' ') for attr in attribute_list]
attr_value_dict = {k.replace('_', ' '):v for k,v in attribute_value_dict.items()}
attr_act_corl_mat = json.load(open(os.path.join("attr_act_corl_mat.json")))
attr_act_corl_mat = {k.replace('_', ' '):v for k,v in attr_act_corl_mat.items()}
attr_value_num = json.load(open(os.path.join("attr_value_num.json")))
attr_value_num = {k.replace('_', ' '):v for k,v in attr_value_num.items()}
skin_color_lst = attr_value_num['skin color']
attr_value_num['skin color'] = [skin_color_lst[0], skin_color_lst[1], skin_color_lst[2], skin_color_lst[3], sum(skin_color_lst[-3:])]


# In[2]:


attr_act_corl_skin = attr_act_corl_mat['skin color']

print(len(attr_act_corl_skin[0]))

coexist_corl = [sum(item) for item in zip(attr_act_corl_skin[4], attr_act_corl_skin[5], attr_act_corl_skin[6])]
print(len(coexist_corl))
attr_act_corl_mat['skin color'] = [attr_act_corl_skin[0], attr_act_corl_skin[1], attr_act_corl_skin[2], 
                                   attr_act_corl_skin[3], coexist_corl]

fontsize_ticks = 24
fontsize_legend = 36
fontsize_axis_labels = 48


# In[3]:


def plot_action_attribute_correlation():
    attr_value_dict = {'skin color': ['unidentifiable', 'white', 'brown/yellow', 'black', 'coexisting'], 
                   'relationship': ['unidentifiable', 'identifiable'], 
                   'face': ['invisible', 'partially visible', 'completely visible'], 
                   'nudity': ['no-nudity', 'partial-nudity', 'semi-nudity'], 
                   'gender': ['unidentifiable', 'male', 'female', 'coexisting']}
    
    plt.figure(figsize=(48, 36))
    plt.subplots_adjust(left=0.05, bottom=0, right=1.1, top=1, wspace=0.1, hspace=0.05)
    for i, attr in enumerate(attr_list):
        print(len(attr_act_corl_mat[attr][0]))
        attr_act_corl_chunk = np.array(attr_act_corl_mat[attr])
        norm_chunk = np.reciprocal(np.tile(np.sum(attr_act_corl_chunk, axis=0), (attr_act_corl_chunk.shape[0],1)))
        heat_matrix = np.multiply(attr_act_corl_chunk, norm_chunk)
        
        # fig, ax = plt.subplots()
        ax = plt.subplot(5,1,i+1)
        
        plt.imshow(heat_matrix, cmap='hot', interpolation='nearest')
        
#         from mpl_toolkits.axes_grid1 import make_axes_locatable
#         divider = make_axes_locatable(ax)
#         cax = divider.append_axes("right", size="2%", pad=1.2)
#         cbar = plt.colorbar(orientation='vertical', shrink=0.8, cax=cax)
        cbar = plt.colorbar(orientation='vertical', shrink=0.8)
        cbar.ax.tick_params(labelsize=48)
        #
        # We want to show all ticks...
        ax.set_xticks(np.arange(len(action_list)))
        ax.set_yticks(np.arange(len(attr_value_dict[attr])))
        # ... and label them with the respective list entries
        ax.set_xticklabels([fname.replace('_', ' ') for fname in action_list])
        ax.set_yticklabels(attr_value_dict[attr])
        ax.tick_params(labelsize=fontsize_axis_labels, which='both', axis='both', length=0)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
#         plt.xticks(fontsize=fontsize_ticks, weight='medium')
#         plt.yticks(fontsize=fontsize_ticks, weight='medium')

        # title, tick, etc.
        plt.title(attr.replace('_', ' '), fontsize=100)
        # plt.tick_params(labelsize=fontsize_ticks)

        # plt.gcf().subplots_adjust(bottom=0.05, top=0.98, left=0.05, right=0.95)

    plt.savefig('action_attribute_correlation_details.pdf',bbox_inches="tight", pad_inches=0)   


# In[4]:


def plot_action_attribute_correlation_transposed():
    matplotlib.rcParams['xtick.major.pad'] = 10
    plt.figure(figsize=(14, 24))
    ax = plt.subplot(1,1,1)
    heat_matrix_lst = []
    attr_value_lst = []
    for i, attribute in enumerate(attr_list):
        attr_act_corl_chunk = np.array(attr_act_corl_mat[attribute])
        norm_chunk = np.reciprocal(np.tile(np.sum(attr_act_corl_chunk, axis=0), (attr_act_corl_chunk.shape[0],1)))
        heat_matrix_lst.append(np.multiply(attr_act_corl_chunk, norm_chunk))
        attr_value_lst.extend(attr_value_dict[attribute])
        #print(np.array(attr_act_corl_mat[attribute]).shape)

#         print(np.tile(np.sum(attr_act_corl_chunk, axis=0), (attr_act_corl_chunk.shape[0],1)))
#         print(attr_act_corl_chunk.shape)
#         print(np.sum(np.array(attr_act_corl_mat[attribute]), axis=0))
#         print(np.sum(np.array(attr_act_corl_mat[attribute]), axis=1))
#         print(np.sum(np.array(attr_act_corl_mat[attribute])))

    heat_matrix = np.transpose(np.vstack(heat_matrix_lst))
        
    img = ax.imshow(heat_matrix, cmap='hot', interpolation='nearest')
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(img, ax=ax, orientation='vertical', shrink=0.8, cax=cax)
    
    # We want to show all ticks...
    ax.set_yticks(np.arange(len(action_list)))
    ax.set_xticks(np.arange(len(attr_value_lst)))
    ax.xaxis.labelpad = 10

    
    # ... and label them with the respective list entries
    ax.set_yticklabels([fname.replace('_', ' ') for fname in action_list])
    ax.set_xticklabels(attr_value_lst)

    
    plt.tick_params(labelsize=fontsize_ticks)
    plt.xticks(fontsize=fontsize_ticks, weight='medium')
    plt.yticks(fontsize=fontsize_ticks, weight='medium')
    ax.tick_params(labelsize=fontsize_ticks)
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center", rotation_mode="anchor")
    
    
    def bracket(ax, len_br, text, pos=[0,0], scalex=1, scaley=1, linekw = {}, textkw = {}):    
        x = np.array([0, 0.05*len_br, 0.45*len_br,0.5*len_br])
        y = np.array([0,-0.02,-0.02,-0.025])
        x = np.concatenate((x,x+0.5*len_br)) 
        y = np.concatenate((y,y[::-1]))
        ax.plot(x*scalex+pos[0], y*scaley+pos[1], clip_on=False, transform=ax.get_xaxis_transform(), **linekw)
        print(pos[0]+0.5*scalex+0.5*len_br)
        ax.text(pos[0]+0.5*len_br, y.min()*scaley+pos[1], text, 
                transform=ax.get_xaxis_transform(),
                ha="center", va="top", **textkw)
        
    len_br_lst = [5,2,3,3,4]
    text_lst = ['skin color', 'relationship', 'face', 'nudity', 'gender']
    sum_len = 0
    for i in range(5):
        len_br = len_br_lst[i]
        text = text_lst[i]
        bracket(ax, len_br, text, pos=[-0.5+sum_len,0], scalex=1, scaley=1, 
                linekw=dict(color="black", ls='-', lw=2), textkw=dict(color="blue", fontweight='bold', fontsize=24))
        sum_len += len_br

    plt.autoscale(tight=True)
    plt.tight_layout()
    plt.savefig('action_attribute_correlation.pdf',bbox_inches="tight", pad_inches=0)   


# In[5]:


print(attr_value_dict)


# In[6]:


if __name__ == '__main__':
    plot_action_attribute_correlation()
#     plot_action_attribute_correlation_transposed()

