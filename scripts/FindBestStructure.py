'''
Main script for optimizing ORR catalyst structure
Two stage optimization
Stage 1: Maximize current density and minimize surface energy simultaneously
Stage 2: Minimize surface energy by moving atoms to adjacent locations at constant loading
'''

import os
import numpy as np
import random
from orr_cat import orr_cat
from sim_anneal import *


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

max_ind = np.argmax(quenched_data[:,1])

print 'The best structure is in ' + subfldr_list[max_ind]