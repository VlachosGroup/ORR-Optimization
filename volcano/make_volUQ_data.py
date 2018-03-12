'''
1. Plots a heat map of binding energy errors
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

#for metal_name in ['Pt','Au']:
metal_name = 'Pt'
x = metal(metal_name)

n_GCNs = 100

if metal_name == 'Pt':
    GCN_vec = np.linspace(1,10,n_GCNs)
elif metal_name == 'Au':
    GCN_vec = np.linspace(2,8,n_GCNs)

det_vol = np.zeros(n_GCNs)
UQ_uncorr_vol = np.zeros(n_GCNs)
UQ_corr_vol = np.zeros(n_GCNs)
UQ_corr_vol_upper = np.zeros(n_GCNs)
UQ_corr_vol_lower = np.zeros(n_GCNs)
    
for i in xrange(n_GCNs):

    print 'GCN: ' + str(GCN_vec[i])
    
    # Data for deterministic volcano
    BEs = x.get_BEs(GCN_vec[i], uncertainty = False)
    rate = ORR_rate(BEs[0], BEs[1],explicit=False)
    det_vol[i] = rate
    
    n_MC_samples = 1000
    data_uncorr = np.zeros(n_MC_samples)
    data_corr = np.zeros(n_MC_samples)
    
    for j in xrange(n_MC_samples):

        # Unocorrelated data
        BEs = x.get_BEs(GCN_vec[i], uncertainty = True, correlations = False)
        rate = ORR_rate(BEs[0], BEs[1],explicit=False)
        data_uncorr[j] = rate
    
        # Correlated data
        BEs = x.get_BEs(GCN_vec[i], uncertainty = True, correlations = True)
        rate = ORR_rate(BEs[0], BEs[1],explicit=False)
        data_corr[j] = rate
    
    data_uncorr = np.sort(data_uncorr)
    data_corr = np.sort(data_corr)
    
    UQ_uncorr_vol[i] = np.exp( np.mean( np.log(data_uncorr)))
    UQ_corr_vol[i] = np.exp( np.mean( np.log(data_corr)))

'''
Write data to numpy files
'''
    
    
'''
Plot volcano plots
'''
    
fig = plt.figure()
plt.plot(GCN_vec, det_vol, label = 'deterministic')
plt.plot(GCN_vec, UQ_corr_vol, label = 'correlated UQ')
plt.plot(GCN_vec, UQ_uncorr_vol, label = 'uncorrelated UQ')
plt.xticks(size=20)
plt.yticks(size=20)
plt.xlabel('GCN',size=20)
plt.ylabel('log(rate)',size=20)
plt.yscale('log')
plt.legend(loc=4, prop={'size':20}, frameon=False)
plt.tight_layout()
plt.savefig(metal_name + '_UQ_volcano.png', format='png', dpi=600)
plt.close()