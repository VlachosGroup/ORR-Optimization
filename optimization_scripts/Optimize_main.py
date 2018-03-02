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

if len(sys.argv) > 1:
    MOO_weight = float(sys.argv[1]) / 1000.0 # divide by total number of jobs
    MOO_weight = MOO_weight ** 2
    data_fldr = str(np.int(sys.argv[1]) + 1000)
    random.seed(a=int(sys.argv[1]))
else:
    MOO_weight = 1.0
    data_fldr = 'solo_run'

# Make folder for new structure
if os.path.exists(data_fldr ):
    # Clear folder contents
    for the_file in os.listdir(data_fldr ):
        file_path = os.path.join(data_fldr , the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(e)
else:
    os.makedirs(data_fldr )

print MOO_weight
np.save(os.path.join(data_fldr, 'omega.npy'), MOO_weight)

'''
Build catalyst structure
'''

cat = orr_cat(dim1 = 20, dim2 = 20, volcano = 'CV')
cat.randomize(build_structure = True)

# Compute normalization factors
E_form_norm = cat.metal.E_coh / 12.0
I_norm = 0.5 * cat.i_max

# Show structure
cat.show(fname = os.path.join(data_fldr, 'pre_optimized'), fmat = 'png', transmute_top = True)
cat.show(fname = os.path.join(data_fldr, 'pre_optimized'), fmat = 'xsd', transmute_top = True)

'''
Multiobjective optimization (stage 1)
'''

# Optimize
traj_hist_a = optimize(cat, weight = MOO_weight, ensemble = 'GCE', n_cycles = 500, T_0 = 1.2, 
    j_norm = I_norm, se_norm = E_form_norm, n_record = 100, verbose = True)
np.save(os.path.join(data_fldr, 'trajectory_a.npy'), traj_hist_a)

mat.rcParams['mathtext.default'] = 'regular'
mat.rcParams['text.latex.unicode'] = 'False'
mat.rcParams['legend.numpoints'] = 1
mat.rcParams['lines.linewidth'] = 4
mat.rcParams['lines.markersize'] = 12

# Plot current density profile
fig = plt.figure()
plt.plot(traj_hist_a[:,0], traj_hist_a[:,1], '-')
plt.xticks(size=20)
plt.yticks(size=20)
plt.xlabel('Metropolis steps',size=20)
plt.ylabel('Current density (mA/cm^2)',size=20)
plt.tight_layout()
plt.savefig(os.path.join(data_fldr, 'traj_j_a.png'), format='png', dpi=600)
plt.close()

# Plot surface energy profile
fig = plt.figure()
plt.plot(traj_hist_a[:,0], traj_hist_a[:,2], '-')
plt.xticks(size=20)
plt.yticks(size=20)
plt.xlabel('Metropolis steps',size=20)
plt.ylabel('Surface energy (J/m^2)',size=20)
plt.tight_layout()
plt.savefig(os.path.join(data_fldr, 'traj_se_a.png'), format='png', dpi=600)
plt.close()

# Show structure
cat.occs_to_atoms()
cat.show(fname = os.path.join(data_fldr, 'optimized'), fmat = 'png', transmute_top = True)
cat.show(fname = os.path.join(data_fldr, 'optimized'), fmat = 'xsd', transmute_top = True)

'''
Surface energy minimization to meta-stable structure (stage 2)
'''

# Optimize
traj_hist_b = optimize(cat, weight = 0., ensemble = 'CE', n_cycles = 25, T_0 = 0, n_record = 100, verbose = True)
np.save(os.path.join(data_fldr, 'trajectory_b.npy'), traj_hist_b)

# Plot current density profile
fig = plt.figure()
plt.plot(traj_hist_b[:,0], traj_hist_b[:,1], '-')
plt.xticks(size=20)
plt.yticks(size=20)
plt.xlabel('Metropolis steps',size=20)
plt.ylabel('Current density (mA/cm^2)',size=20)
plt.tight_layout()
plt.savefig(os.path.join(data_fldr, 'traj_j_b.png'), format='png', dpi=600)
plt.close()

# Plot surface energy profile
fig = plt.figure()
plt.plot(traj_hist_b[:,0], traj_hist_b[:,2], '-')
plt.xticks(size=20)
plt.yticks(size=20)
plt.xlabel('Metropolis steps',size=20)
plt.ylabel('Surface energy (J/m^2)',size=20)
plt.tight_layout()
plt.savefig(os.path.join(data_fldr, 'traj_se_b.png'), format='png', dpi=600)
plt.close()

# Show structure
cat.occs_to_atoms()
cat.show(fname = os.path.join(data_fldr, 'meta_stable'), fmat = 'png', transmute_top = True)
cat.show(fname = os.path.join(data_fldr, 'meta_stable'), fmat = 'xsd', transmute_top = True)