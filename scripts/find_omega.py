'''
Main script for optimizing ORR catalyst structure
Two stage optimization
Stage 1: Maximize current density and minimize surface energy simultaneously
Stage 2: Minimize surface energy by moving atoms to adjacent locations at constant loading
'''

import os
import numpy as np
import random
from from orr_optimizer.orr_cat import *orr_cat import orr_cat
from from orr_optimizer.orr_cat import *sim_anneal import *

import matplotlib.pyplot as plt
import matplotlib as mat

fldr = '.'

# List all folders
subfldr_list = []
for fldr_or_file in os.listdir(fldr):
    if os.path.isdir(os.path.join(fldr,fldr_or_file)):
        subfldr_list.append(fldr_or_file)
        

for subfldr in subfldr_list:
    ind = np.float(subfldr)
    omega = ind / 1000.0
    np.save(os.path.join(subfldr, 'omega.npy'), omega)