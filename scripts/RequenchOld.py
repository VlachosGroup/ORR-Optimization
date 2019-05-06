'''
Reads Pareto optimizations produced by the old Matlab code
'''

import os
import sys
sys.path.append('/home/vlachos/mpnunez/Github/Zacros-Wrapper')
sys.path.append('/home/vlachos/mpnunez/Github/Structure-Optimization')
sys.path.append('/home/vlachos/mpnunez/python_packages/ase')
sys.path.append('/home/vlachos/mpnunez/python_packages/sklearn/lib/python2.7/site-packages')
this_folder = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(this_folder,'..','structure')) 
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
sys.setrecursionlimit(1500)
cat = orr_cat(dim1 = 30, dim2 = 30, volcano = 'CV')

unquenched_data = []
quenched_data = []
re_quenched_data = []
for subfldr in subfldr_list:
    
    # Unquenched structure
    f = open(os.path.join(subfldr, "optimum.bin"), "r")
    a = np.fromfile(f, dtype=np.uint32)
    a = a[2700::]       # ignore occupancies of first 3 layers of 900 atoms each
    cat.assign_occs(a)
    
    j = cat.eval_current_density(normalize = True)
    seng = cat.eval_surface_energy(normalize = True)
    unquenched_data.append([j, seng])
    print [j, seng]
    
    # Structure quenched with Matlab
    f = open(os.path.join(subfldr, 'quench', "optimum.bin"), "r")
    a = np.fromfile(f, dtype=np.uint32)
    a = a[2700::]
    cat.assign_occs(a)

    j = cat.eval_current_density(normalize = True)
    seng = cat.eval_surface_energy(normalize = True)
    quenched_data.append([j, seng])
    print [j, seng]
    
    # Structure requenched with Python
    
    
    if j > 40.0:
        traj_hist_b = optimize(cat, weight = 0., ensemble = 'CE', n_cycles = 25,
            T_0 = 0, n_record = 100, verbose = True)
            
    j = cat.eval_current_density(normalize = True)
    seng = cat.eval_surface_energy(normalize = True)
    re_quenched_data.append([j, seng])
    print [j, seng]
 
unquenched_data = np.array(unquenched_data)
quenched_data = np.array(quenched_data)
re_quenched_data = np.array(re_quenched_data)

np.save('unquenched_data.npy',unquenched_data)
np.save('quenched_data.npy',quenched_data)
np.save('re_quenched_data.npy',re_quenched_data)

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
plt.plot( re_quenched_data[:,1], re_quenched_data[:,0], 'o', color = 'm', label = 'requenched')
plt.plot( Pareto_front[:,1], Pareto_front[:,0], '-*', color = 'g', label = 'Pareto front')
plt.xticks(size=18)
plt.yticks(size=18)
plt.xlabel(r'Surface energy (J/m$^2$)', size=24)
plt.ylabel(r'Current density (mA/cm$^2$)', size=24)
plt.legend(loc=4, prop={'size':20}, frameon=False)
plt.tight_layout()
plt.savefig(os.path.join(fldr, 'Pareto_plot'), dpi = 600)
plt.close()