'''
Plots a heat map of binding energy errors
'''

import os
import sys
this_folder = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(this_folder,'..','structure')) 

from metal import metal
from ORR import ORR_rate
import numpy as np
from scipy.stats import norm

import matplotlib.pyplot as plt
import matplotlib as mat

metal_name = 'Au'

'''
Read data
'''

all_data = np.load(metal_name + '_UQ_vol.npy')

Na = 6.022e23
mA_to_kA = 1.0e-6

# Convert from [miliAmps per atom] to [kiloAmps per mol]
all_data[1::,:] = all_data[1::,:] * mA_to_kA * Na

GCN_vec = all_data[0,:]
det_vol = all_data[1,:]
UQ_uncorr_vol = all_data[2,:]
UQ_corr_vol = all_data[3,:]
UQ_corr_vol_upper = all_data[4,:]
UQ_corr_vol_lower = all_data[5,:]

'''
Plot volcano plots
'''

mat.rcParams['mathtext.default'] = 'regular'
mat.rcParams['text.latex.unicode'] = 'False'
mat.rcParams['legend.numpoints'] = 1
mat.rcParams['lines.linewidth'] = 2
mat.rcParams['lines.markersize'] = 12

fig = plt.figure()
ax = plt.gca()
ax.fill_between(GCN_vec, UQ_corr_vol_lower, UQ_corr_vol_upper, facecolor=[1.0, 0.5, 0.5])
plt.plot(GCN_vec, det_vol, label = 'deterministic', color = 'k')
plt.plot(GCN_vec, UQ_uncorr_vol, label = 'uncorrelated', color = 'b')
plt.plot(GCN_vec, UQ_corr_vol, label = 'correlated', color = 'g')
if metal_name == 'Au':
    plt.xlim(3,8)
    plt.ylim(10**-7,10**3)
plt.xticks(size=24)
plt.yticks(size=24)
ax.tick_params(direction='in', length=6, width=2, colors='k')
plt.xlabel('Generalized coordination number',size=24)
plt.ylabel(r'Current, $i$ [kA/mol]', size=24)
plt.yscale('log')
plt.legend(loc=4, prop={'size':24}, frameon=False)
plt.tight_layout()
plt.savefig(metal_name + '_UQ_volcano.png', format='png', dpi=600)
plt.close()