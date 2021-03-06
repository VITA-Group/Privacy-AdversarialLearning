#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import matplotlib.pyplot as plt
plt.style.use([ '/Users/wuzhenyu/.matplotlib/stylelib/science.mplstyle' , '/Users/wuzhenyu/.matplotlib/stylelib/ieee.mplstyle' ])
from matplotlib.lines import Line2D

naive_downsampling_target = np.array([68.9, 66.3, 59.4, 52.7, 48.3]) # action
naive_downsampling_budget = np.array([77.3, 75.3, 73.3, 72.8, 70.9]) # actor
rate = np.array([1, 2, 4, 8, 16])

empirical_obfuscation_target = np.array([66.7, 38.2, 67.4, 41.5, 66.2, 40.8, 67.5, 39.5])
empirical_obfuscation_budget = np.array([74.3, 68.9, 74.8, 70.2, 74.3, 69.2, 75.4, 68.6]) 

ours_GRL_target = np.array([62.9])
ours_GRL_budget = np.array([71.4])
ours_GRL_target_r = np.array([63.6])
ours_GRL_budget_r = np.array([70.5])

ours_entropy_target = np.array([64.7, 64.0, 64.9, 64.3]) 
ours_entropy_budget = np.array([65.2, 65.9, 67.7, 67.5])
ours_entropy_target_r = np.array([66.6, 67.5, 65.3, 66.4]) 
ours_entropy_budget_r = np.array([64.6, 64.7, 67.3, 66.0])
_M = np.array([8,4,2,1])

ours_kbeam_target = np.array([64.8, 66.3, 62.2, 64.4])
ours_kbeam_budget = np.array([72.5, 72.0, 69.5, 71.0])
ours_kbeam_target_r = np.array([65.4, 66.4, 65.7, 63.5])
ours_kbeam_budget_r = np.array([72.2, 71.3, 70.2, 69.3])
_K = np.array([1,2,4,8])
mask = np.array([True, True, True, True])

fontsize_label=10
fontsize_axis=8
fontsize_legend=4.5
markersize_scatter=32
markersize_legend=4
linewidth=1

fig = plt.figure()
plt.grid(linestyle='--')
plt.tick_params(labelsize=fontsize_axis)

plt.scatter(naive_downsampling_budget, naive_downsampling_target, c='grey', marker='o', s=markersize_scatter*np.sqrt(rate/10))
plt.scatter(empirical_obfuscation_budget, empirical_obfuscation_target, c='k', marker='o', s=markersize_scatter)
plt.scatter(ours_kbeam_budget[mask], ours_kbeam_target[mask], c='lime', marker='s', s=markersize_scatter*np.sqrt(_K))
plt.scatter(ours_kbeam_budget_r[mask], ours_kbeam_target_r[mask], c='green', marker='s', s=markersize_scatter*np.sqrt(_K))
plt.scatter(ours_GRL_budget, ours_GRL_target, c='skyblue', marker='^', s=markersize_scatter)
plt.scatter(ours_GRL_budget_r, ours_GRL_target_r, c='blue', marker='^', s=markersize_scatter)
plt.scatter(ours_entropy_budget, ours_entropy_target, c='orange', marker='p', s=markersize_scatter*np.sqrt(_M))
plt.scatter(ours_entropy_budget_r, ours_entropy_target_r, c='red', marker='p', s=markersize_scatter*np.sqrt(_M))

xlim_a, xlim_b = 62.5, 77.5
ylim_a, ylim_b = 25, 70

x = np.linspace(xlim_a,xlim_b,100)
plt.plot(x, x, '--', linewidth=linewidth, c='k')
x = np.linspace(xlim_a,xlim_b,100)
plt.plot(x, [naive_downsampling_target[0]]*100, '--', linewidth=linewidth, c='m')
y = np.linspace(ylim_a,ylim_b,100)
plt.plot([naive_downsampling_budget[0]]*100, y, '--', linewidth=linewidth, c='m')

plt.axis('tight')
plt.xlim(xlim_a,xlim_b)
plt.ylim(ylim_a,ylim_b)
# plt.gca().set_aspect('equal', adjustable='box')
plt.gcf().subplots_adjust(bottom=0.14, top=0.98, left=0.15, right=0.95)

plt.xlabel(r'Privacy Attributes cMAP $A_B^N$ (\%)', fontsize=fontsize_label)
plt.ylabel(r'Action accuracy $A_T$ (\%)', fontsize=fontsize_label)
legend_elements = [
    Line2D([0], [0], marker='o', markeredgecolor='grey', markeredgewidth=linewidth, label='Naive Downsample', markerfacecolor='grey', linestyle='None', markersize=markersize_legend),
    Line2D([0], [0], marker='o', markeredgecolor='k', markeredgewidth=linewidth, label='Empirical Obfuscation ', markerfacecolor='k', linestyle='None', markersize=markersize_legend),
    Line2D([0], [0], marker='^', markeredgecolor='skyblue', markeredgewidth=linewidth, label=r'GRL', markerfacecolor='skyblue', linestyle='None', markersize=markersize_legend),
    Line2D([0], [0], marker='^', markeredgecolor='blue', markeredgewidth=linewidth, label=r'GRL$^{+}$', markerfacecolor='blue', linestyle='None', markersize=markersize_legend),
    Line2D([0], [0], marker='s', markeredgecolor='lime', markeredgewidth=linewidth, label=r'Ours-$K$-Beam', markerfacecolor='lime', linestyle='None', markersize=markersize_legend),
    Line2D([0], [0], marker='s', markeredgecolor='green', markeredgewidth=linewidth, label=r'Ours-$K$-Beam$^{+}$', markerfacecolor='green', linestyle='None', markersize=markersize_legend),
    Line2D([0], [0], marker='p', markeredgecolor='orange', markeredgewidth=linewidth, label=r'Ours-Entropy', markerfacecolor='orange', linestyle='None', markersize=markersize_legend),
    Line2D([0], [0], marker='p', markeredgecolor='red', markeredgewidth=linewidth, label=r'Ours-Entropy$^{+}$', markerfacecolor='red', linestyle='None', markersize=markersize_legend),    
]
leg=plt.legend(
    handles=legend_elements, 
    loc='lower right', fontsize=fontsize_legend, ncol=2, columnspacing=0.5, labelspacing=1.0, frameon=True
    )
leg.get_frame().set_edgecolor('k')
leg.get_frame().set_linewidth(0.5)
plt.savefig('HMDB51.pdf', bbox_inches='tight', pad_inches=0)
plt.close()

