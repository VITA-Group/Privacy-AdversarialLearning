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
               '/Users/wuzhenyu/.matplotlib/stylelib/ieee.mplstyle'
              
              ])
import json, os, csv

from calculate_statistics import attribute_list, attribute_value_dict

attribute_value_dict = {
    'skin_color': ['0', '1', '2', '3', '4'],
    'relationship': ['0', '1'],
    'face': ['0', '1', '2'], 
    'nudity': ['0', '1' ,'2'],
    'gender': ['0', '1', '2', '3']
    }

# filenames = [
#     'brush_hair', 'cartwheel', 'catch', 'chew', 'clap', 'climb', 'climb_stairs', 'dive', 
#     'draw_sword', 'dribble', 'drink', 'eat', 'fall_floor', 'fencing', 'flic_flac', 'golf', 
#     'handstand', 'hit', 'hug', 'jump', 'kick', 'kick_ball', 'kiss', 'laugh', 'pick', 'pour', 
#     'pullup', 'punch', 'push', 'pushup', 'ride_bike', 'ride_horse', 'run', 'shake_hands', 
#     'shoot_ball', 'shoot_bow', 'shoot_gun', 'sit', 'situp', 'smile', 'smoke', 'somersault', 
#     'stand', 'swing_baseball', 'sword', 'sword_exercise', 'talk', 'throw', 'turn', 'walk', 'wave'
# ]

attr_list = [attr.replace('_', ' ') for attr in attribute_list]
attr_value_dict = {k.replace('_', ' '):v for k,v in attribute_value_dict.items()}
attr_act_corl_mat = json.load(open(os.path.join("attr_act_corl_mat.json")))
attr_act_corl_mat = {k.replace('_', ' '):v for k,v in attr_act_corl_mat.items()}
attr_value_num = json.load(open(os.path.join("attr_value_num.json")))
attr_value_num = {k.replace('_', ' '):v for k,v in attr_value_num.items()}

skin_color_lst = attr_value_num['skin color']
attr_value_num['skin color'] = [skin_color_lst[0], skin_color_lst[1], skin_color_lst[2], skin_color_lst[3], sum(skin_color_lst[-3:])]

attr_act_corl_skin = attr_act_corl_mat['skin color']


coexist_corl = [sum(item) for item in zip(attr_act_corl_skin[4], attr_act_corl_skin[5], attr_act_corl_skin[6])]
attr_act_corl_mat['skin color'] = [attr_act_corl_skin[0], attr_act_corl_skin[1], attr_act_corl_skin[2], 
                                   attr_act_corl_skin[3], coexist_corl]

r = csv.reader(open('act_vid_dist.csv', 'r'), delimiter=',')
for i, row in enumerate(r):
    if i == 0:
        action_list = row
    elif i == 1:
        action_number = np.array(row).astype(int)
    print(row)

# action_list = ['DrawSword', 'Kiss', 'RideBike', 'Dribble', 'Hug', 'Dive', 'Cartwheel', 'Drink', 'Climb', 'Eat', 
#                'Walk', 'FallFloor', 'Stand', 'Sword', 'Pick', 'SwingBaseb', 'Handstand', 'ShootBow', 'Fencing', 
#                'Talk', 'ShootBall', 'KickBall', 'Push', 'Pour', 'Wave', 'RideHorse', 'FlicFlac', 'Turn', 'ShakeHands',
#                'Somersault', 'ClimbStairs', 'Golf', 'Smoke', 'Situp', 'ShootGun', 'Punch', 'Run', 'Sit', 'Smile', 
#                'Kick', 'SwordExer', 'Laugh', 'Throw', 'Pushup', 'Hit', 'Chew', 'Clap', 'Pullup', 'Catch', 'Jump',
#                'BrushHair']


heat_matrix_lst = []
attr_value_lst = []
for i, attr in enumerate(attr_list):
    attr_act_corl_chunk = np.array(attr_act_corl_mat[attr])
    norm_chunk = np.reciprocal(np.tile(np.sum(attr_act_corl_chunk, axis=0), (attr_act_corl_chunk.shape[0],1)))
    heat_matrix_lst.append(np.multiply(attr_act_corl_chunk, norm_chunk))
    attr_value_lst.extend(attr_value_dict[attr])

heat_matrix = np.transpose(np.vstack(heat_matrix_lst))


fontsize_ticks = 54
fontsize_legend = 36
fontsize_axis_labels = 36
print(action_list)


# In[2]:


'''
Plot action - Number of Frames (i.e., Fig 8 in camera ready version)
'''
def plot_action_attributes_sidebyside_v1(action_list, action_number, heat_matrix):

    plt.figure(figsize=(48, 24))
    
    axes = [plt.subplot(1,2,i+1) for i in range(2)]
    #fig, axes = plt.subplots(1, 2, sharey=True)
    #fig.subplots_adjust(wspace=0, hspace=0)

    y_pos = np.arange(len(action_list))
    
    axes[0].barh(y_pos, action_number, align='center', color='b')
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(action_list)
    axes[0].invert_yaxis()
    
    #plt.tick_params(labelsize=fontsize_ticks)
    #plt.xticks(fontsize=fontsize_ticks, weight='medium')
    #plt.yticks(fontsize=fontsize_ticks, weight='medium')
    axes[0].tick_params(labelsize=fontsize_ticks)
        
    img = axes[1].imshow(heat_matrix, cmap='hot', interpolation='nearest')
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(axes[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(img, ax=axes[1], orientation='vertical', shrink=0.8, cax=cax)
    
    # We want to show all ticks...
    #ax.set_yticks(np.arange(len(filenames)))
    axes[1].set_xticks(np.arange(len(attr_value_lst)))
    # ... and label them with the respective list entries
    #ax.set_yticklabels([fname.replace('_', ' ') for fname in filenames])
    axes[1].set_xticklabels(attr_value_lst)
    
    #plt.xticks(fontsize=fontsize_ticks, weight='medium')
    #plt.yticks(fontsize=fontsize_ticks, weight='medium')
    axes[1].tick_params(labelsize=fontsize_ticks)
    # Rotate the tick labels and set their alignment.
    plt.setp(axes[1].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    plt.autoscale(tight=True)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig('action_attribute_sidebyside.pdf',bbox_inches="tight", pad_inches=0)   


# In[3]:


def plot_action_attributes_sidebyside_v2(action_list, action_number, heat_matrix):    

    fig = plt.figure(figsize=(36, 24))
    
    #fig, axes = plt.subplots(1, 2, sharey=True)
    fig.subplots_adjust(wspace=0, hspace=0)

    ax1 = fig.add_subplot(121)
    
    y_pos = np.arange(len(action_list))
    
    ax1.barh(y_pos, action_number, align='center', color='b')
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(action_list)
    ax1.invert_yaxis()
    
    #plt.tick_params(labelsize=fontsize_ticks)
    plt.xticks(fontsize=fontsize_ticks, weight='medium')
    plt.yticks(fontsize=fontsize_ticks, weight='medium')
    ax1.tick_params(labelsize=fontsize_ticks)
    font = {'family': 'Times New Roman',
            'color':  'black',
            'weight': 'semibold',
            'size': fontsize_axis_labels,
    }
    # ax.set_ylabel('Actions', fontdict=font)
    # ax.set_xlabel('Number of Videos', fontdict=font)
    plt.autoscale(tight=True)
    plt.tight_layout()
    #plt.gcf().subplots_adjust(bottom=0.15, top=0.85, left=0.15, right=0.85)

    
    ax2 = fig.add_subplot(122)

    img =  ax2.imshow(heat_matrix, cmap='hot', interpolation='nearest')
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(img, ax=ax2, orientation='vertical', shrink=0.8, cax=cax)
    
    # We want to show all ticks...
    #ax.set_yticks(np.arange(len(filenames)))
    ax2.set_xticks(np.arange(len(attr_value_lst)))
    # ... and label them with the respective list entries
    #ax.set_yticklabels([fname.replace('_', ' ') for fname in filenames])
    ax2.set_xticklabels(attr_value_lst)
    
    plt.tick_params(labelsize=fontsize_ticks)
    plt.xticks(fontsize=fontsize_ticks, weight='medium')
    plt.yticks(fontsize=fontsize_ticks, weight='medium')
    ax2.tick_params(labelsize=fontsize_ticks)
    # Rotate the tick labels and set their alignment.
    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    fig.subplots_adjust(wspace=0, hspace=0)

    plt.autoscale(tight=True)
    plt.tight_layout()
    plt.savefig('action_attribute_sidebyside.pdf',bbox_inches="tight", pad_inches=0)   


# In[4]:


def plot_action_attributes_sidebyside_v3(action_list, action_number, heat_matrix):    
    matplotlib.rcParams['xtick.major.pad'] = 15

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(40, 44))
    
    fig.subplots_adjust(wspace=0, hspace=0)

    #ax1 = fig.add_subplot(121)
    
    y_pos = np.arange(len(action_list))
    
    ax1.barh(y_pos, action_number, height=0.8, align='center', color='b')
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(action_list)
    ax1.invert_yaxis()
    
    print(action_list)
    print(action_number)
    
    ax1.tick_params(labelsize=fontsize_ticks, which='both', axis='both')
    # ax1.tick_params(labelsize=fontsize_ticks, which='both', axis='x')

    ax1.yaxis.set_ticks_position('left')
    ax1.xaxis.set_ticks_position('bottom')
    ax1.tick_params(axis='x', which='both', bottom=True)
    ax1.tick_params(axis='y', which='both', left=False)
    
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    
    
    ax1.set_ylim(len(action_list)-0.5, -.5)

    #ax2 = fig.add_subplot(122)
        
    img =  ax2.imshow(heat_matrix, cmap='hot', interpolation='nearest')
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("left", size="5%", pad=1.2)
    cbar = plt.colorbar(img, ax=ax2, orientation='vertical', shrink=0.8, cax=cax)
    cbar.ax.tick_params(labelsize=60) 


    ax2.set_xticks(np.arange(len(attr_value_lst)))
    ax2.set_yticks(y_pos)

    ax2.set_xticklabels(attr_value_lst)
    ax2.set_yticklabels(action_list)

    ax2.tick_params(labelsize=fontsize_ticks, which='both', axis='both')
    plt.setp(ax2.get_xticklabels(), rotation=0, ha="center", rotation_mode="anchor")
    ax2.xaxis.labelpad = 15

    ax2.yaxis.set_ticks_position('right')
    ax2.xaxis.set_ticks_position('bottom')
    ax2.tick_params(axis='y', which='both', right=False)
    ax2.tick_params(axis='x', which='both', bottom=False)
    
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
    text_lst = ['skin color', 'relation', 'face', 'nudity', 'gender']
    sum_len = 0
    for i in range(5):
        len_br = len_br_lst[i]
        text = text_lst[i]
        bracket(ax2, len_br, text, pos=[-0.5+sum_len,0], scalex=1, scaley=1, 
                linekw=dict(color="blue", ls='-', lw=2), textkw=dict(color="blue", fontweight='bold', fontsize=54))
        sum_len += len_br
    
    fig.subplots_adjust(wspace=0, hspace=0)
    
    #plt.autoscale(tight=True)
    plt.tight_layout()
    plt.savefig('action_attribute_sidebyside.pdf',bbox_inches="tight", pad_inches=0)


# In[5]:


def plot_action(action_number, action_list):    
    matplotlib.rcParams['xtick.major.pad'] = 15
    fig, ax = plt.subplots(1, 1, figsize=(24, 60))
    ax = plt.subplot(1,1,1)

    y_pos = np.arange(len(action_list))
    
    ax.barh(y_pos, action_number, height=0.75, align='center', color='b')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(action_list)
    ax.invert_yaxis()

    ax.tick_params(labelsize=fontsize_ticks, which='both', axis='y')
    ax.tick_params(labelsize=fontsize_ticks, which='both', axis='x')
    ax.tick_params(axis='x', which='minor', bottom=True)

    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    plt.autoscale(tight=True)
    plt.tight_layout()
    
    plt.autoscale(tight=True)
    plt.tight_layout()
    plt.savefig('action.pdf',bbox_inches="tight", pad_inches=0)       


# In[6]:


def plot_action_attribute_corl(heat_matrix):    
    matplotlib.rcParams['xtick.major.pad'] = 15
    fig, ax = plt.subplots(1, 1, figsize=(24, 48))
    ax = plt.subplot(1,1,1)
    
    img =  ax.imshow(heat_matrix, cmap='hot', interpolation='nearest')
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    #cax = divider.append_axes("right", size="5%", pad=0.8)
    cax = divider.append_axes("right", size="5%", pad=1.0)
    cbar = plt.colorbar(img, ax=ax, orientation='vertical', shrink=0.8, cax=cax)
    cbar.ax.tick_params(labelsize=fontsize_ticks) 

    # We want to show all ticks...
    y_pos = np.arange(len(action_list))
    ax.set_xticks(np.arange(len(attr_value_lst)))
    ax.set_yticks(y_pos)
    
    ax.set_xticklabels(attr_value_lst)
    ax.set_yticklabels(action_list)

    ax.tick_params(labelsize=fontsize_ticks, which='both', axis='both')
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax.xaxis.labelpad = 15
    
    # Hide the right and top spines
#     ax2.spines['right'].set_visible(False)
#     ax2.spines['top'].set_visible(False)

    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.tick_params(axis='y', which='both', right=False)
    ax.tick_params(axis='x', which='both', bottom=False)

    def bracket(ax, len_br, pos=[0,0], scalex=1, scaley=1, linekw = {}):    
        x = np.array([0, 0.05*len_br, 0.45*len_br,0.5*len_br])
        y = np.array([0,-0.005,-0.005,-0.01])
        x = np.concatenate((x,x+0.5*len_br)) 
        y = np.concatenate((y,y[::-1]))
        ax.plot(x*scalex+pos[0], y*scaley+pos[1], clip_on=False, transform=ax.get_xaxis_transform(), **linekw)

    len_br_lst = [4,2,3,3,3]
    sum_len = 0
    for len_br in len_br_lst:
        bracket(ax, len_br, pos=[-0.5+sum_len,0], scalex=1, scaley=1, linekw=dict(color="blue", ls='-', lw=2))
        sum_len += len_br
    
    fig.subplots_adjust(wspace=0, hspace=0)
    
    plt.autoscale(tight=True)
    plt.tight_layout()
    plt.savefig('action_attribute_correlation.pdf',bbox_inches="tight", pad_inches=0)


# In[7]:


def plot_action_attributes_sidebyside_demo(action_list, action_number):
    def bar_plot(action_list, action_number, ax):
        y_pos = np.arange(len(action_list))

        ax.barh(y_pos, action_number, height=0.75, align='center', color='b')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(action_list)

        ax.invert_yaxis()

        ax.tick_params(labelsize=12, which='both', axis='both')

        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')

        ax.set_ylim(len(action_list)-0.5, -.5)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 10))
    fig.subplots_adjust(wspace=0, hspace=0)
    # ax1 = fig.add_subplot(121)
    bar_plot(action_list, action_number, ax1)
    # ax2 = fig.add_subplot(122)
    bar_plot(action_list, action_number, ax2)
    plt.tight_layout()
    plt.savefig('action_attribute_sidebyside.pdf',bbox_inches="tight", pad_inches=0)


# In[8]:


if __name__ == '__main__':
    plot_action_attributes_sidebyside_v3(action_list, action_number, heat_matrix)

