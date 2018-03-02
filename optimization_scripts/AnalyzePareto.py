'''
Main script for optimizing ORR catalyst structure
Two stage optimization
Stage 1: Maximize current density and minimize surface energy simultaneously
Stage 2: Minimize surface energy by moving atoms to adjacent locations at constant loading
'''

import sys
sys.path.append('/home/vlachos/mpnunez/Github/ORR-Optimization')

import os
import numpy as np
import random
from orr_cat import orr_cat
from sim_anneal import *

import matplotlib.pyplot as plt
import matplotlib as mat

def is_pareto_efficient(costs):
    """
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python
    """
    is_efficient = np.ones(costs.shape[0], dtype = bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient]<=c, axis=1)  # Remove dominated points
    return is_efficient

fldr = '.'

# List all folders
subfldr_list = []
for fldr_or_file in os.listdir(fldr):
    if os.path.isdir(os.path.join(fldr,fldr_or_file)):
        subfldr_list.append(fldr_or_file)
        

'''
Build catalyst structure
'''
unquenched_data = []
quenched_data = []
for subfldr in subfldr_list:
    traj_a = np.load(os.path.join(subfldr, 'trajectory_a.npy'))
    traj_b = np.load(os.path.join(subfldr, 'trajectory_b.npy'))
    unquenched_data.append(traj_a[-1,1:3:])
    quenched_data.append(traj_b[-1,1:3:])
 
unquenched_data = np.array(unquenched_data)
quenched_data = np.array(quenched_data) 

# Find Pareto optimal structures
x = np.transpose(np.vstack([-quenched_data[:,0], quenched_data[:,1] ]))
is_PE = is_pareto_efficient(x)
Pareto_front = quenched_data[is_PE,:]
Pareto_front = Pareto_front[Pareto_front[:,1].argsort()]

'''
Pareto plot
'''

mat.rcParams['mathtext.default'] = 'regular'
mat.rcParams['text.latex.unicode'] = 'False'
mat.rcParams['legend.numpoints'] = 1
mat.rcParams['lines.linewidth'] = 2
mat.rcParams['lines.markersize'] = 12

plt.figure()
plt.plot( unquenched_data[:,1], unquenched_data[:,0], 'o', color = 'r', label = 'unquenched')
plt.plot( quenched_data[:,1], quenched_data[:,0], 'o', color = 'b', label = 'quenched')
plt.plot( Pareto_front[:,1], Pareto_front[:,0], '-*', color = 'g', label = 'Pareto front')
plt.xticks(size=18)
plt.yticks(size=18)
plt.xlabel(r'Surface energy (J/m$^2$)', size=24)
plt.ylabel(r'Current density (mA/cm$^2$)', size=24)
plt.legend(loc=4, prop={'size':20}, frameon=False)
plt.tight_layout()
plt.savefig(os.path.join(fldr, 'Pareto_plot'), dpi = 600)
plt.close()