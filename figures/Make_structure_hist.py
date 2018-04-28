'''
Main script for optimizing ORR catalyst structure
Two stage optimization
Stage 1: Maximize current density and minimize surface energy simultaneously
Stage 2: Minimize surface energy by moving atoms to adjacent locations at constant loading
'''

import os
import numpy as np
import random

import sys
this_folder = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(this_folder,'..','structure')) 
from orr_cat import orr_cat
from sim_anneal import *

import matplotlib.pyplot as plt
import matplotlib as mat

fldr1 = '/home/vlachos/mpnunez/ORR_data/CV_Pareto/1863'
fldr2 = '/home/vlachos/mpnunez/ORR_data/JL_Pareto/912'

sys.setrecursionlimit(1500)
cat = orr_cat(volcano = 'CV')
cat.load_defects(os.path.join(fldr1,'meta_stable.xsd'))
get_data1 = cat.get_site_currents(hist_info = True)

gcn_data1 = np.array(get_data1[0])
activity_data = np.array(get_data1[1])

# Also read 2nd folder and compare histograms
cat2 = orr_cat(volcano = 'JL')
cat2.load_defects(os.path.join(fldr2,'meta_stable.xsd'))
get_data2 = cat2.get_site_currents(hist_info = True)
gcn_data2 = np.array(get_data2[0])
activity_data2 = np.array(get_data2[1])

mat.rcParams['mathtext.default'] = 'regular'
mat.rcParams['text.latex.unicode'] = 'False'
mat.rcParams['legend.numpoints'] = 1
mat.rcParams['lines.linewidth'] = 2
mat.rcParams['lines.markersize'] = 12

plt.figure()
plt.plot(gcn_data1,activity_data,'o')
plt.xticks(size=18)
plt.yticks(size=18)
plt.xlabel(r'Generalized coordination number', size=24)
plt.ylabel(r'Activity', size=24)
plt.legend(loc=1, prop={'size':20}, frameon=False)
plt.tight_layout()
plt.savefig('CV.png', dpi = 600)
plt.close()

plt.figure()
plt.plot(gcn_data2,activity_data2,'o')
plt.xticks(size=18)
plt.yticks(size=18)
plt.xlabel(r'Generalized coordination number', size=24)
plt.ylabel(r'Activity', size=24)
plt.legend(loc=1, prop={'size':20}, frameon=False)
plt.tight_layout()
plt.savefig('JL.png', dpi = 600)
plt.close()

plt.figure()
plt.hist( [gcn_data1, gcn_data2], weights=[np.zeros_like(gcn_data1) + 1. / cat.atoms_per_layer, np.zeros_like(gcn_data2) + 1. / cat.atoms_per_layer], label = ['no interactions','with interactions'])
plt.xticks(size=18)
plt.yticks(size=18)
plt.xlabel(r'Generalized coordination number', size=24)
plt.ylabel(r'Frequency', size=24)
plt.legend(loc=1, prop={'size':20}, frameon=False)
plt.tight_layout()
plt.savefig('GCN_histogram_compare.png', dpi = 600)
plt.close()

plt.figure()
plt.hist( [gcn_data1, gcn_data2], weights=[activity_data/np.sum(activity_data), activity_data2/np.sum(activity_data2)], label = ['no interactions','with interactions'])
plt.xticks(size=18)
plt.yticks(size=18)
plt.xlabel(r'Generalized coordination number', size=24)
plt.ylabel(r'Activity contribution', size=24)
plt.legend(loc=2, prop={'size':20}, frameon=False)
plt.tight_layout()
plt.savefig('activity_histogram_compare.png', dpi = 600)
plt.close()