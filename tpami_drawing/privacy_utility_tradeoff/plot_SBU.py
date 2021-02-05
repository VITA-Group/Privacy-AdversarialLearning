#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import matplotlib.pyplot as plt 
plt.style.use([ '/Users/wuzhenyu/.matplotlib/stylelib/science.mplstyle' , '/Users/wuzhenyu/.matplotlib/stylelib/ieee.mplstyle' ])
from matplotlib.lines import Line2D

naive_downsampling_target = np.array([88.8, 87.9, 87.0, 81.9, 79.5, 74.9, 65.1, 64.4, 56.3, 43.5]) # action
naive_downsampling_budget = np.array([99.1, 98.1, 98.1, 99.1, 98.6, 97.6, 97.6, 97.2, 92.9, 73.1]) # actor
rate = np.array([1,2,3,4,6,8,14,16,28,56])

empirical_obfuscation_target = np.array([88.5, 24.6, 88.4, 44.6, 88.3, 58.3, 88.6, 54.2])
empirical_obfuscation_budget = np.array([98.9, 21.4, 98.4, 49.5, 98.5, 43.4, 98.5, 36.3]) 

ours_GRL_target = np.array([80.9]) 
ours_GRL_budget = np.array([85.4])
ours_GRL_target_r = np.array([82.8]) 
ours_GRL_budget_r = np.array([84.3])

ours_entropy_target = np.array([83.2, 84.1, 82.7, 80.8]) 
ours_entropy_budget = np.array([77.6, 74.4, 72.9, 67.2])
ours_entropy_target_r = np.array([81.7, 82.6, 78.0, 82.2]) 
ours_entropy_budget_r = np.array([60.5, 57.9, 54.6, 47.7])
_M = np.array([1,2,4,8])

ours_kbeam_target = np.array([75.5, 80.5, 81.9, 78.2])
ours_kbeam_budget = np.array([87.9, 78.7, 86.4, 82.2])
ours_kbeam_target_r = np.array([78.4, 84.5, 82.6, 78.7])
ours_kbeam_budget_r = np.array([84.9, 80.6, 81.5, 78.5])
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

xlim_a, xlim_b = 20, 100
ylim_a, ylim_b = 20, 90

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

plt.xlabel(r'Actor Pair Accuracy $A_B^N$ (\%)', fontsize=fontsize_label)
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
plt.savefig('SBU.pdf', bbox_inches='tight', pad_inches=0)
plt.close()

