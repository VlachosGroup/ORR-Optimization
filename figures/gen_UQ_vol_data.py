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

from savitzky_golay import *

#for metal_name in ['Pt','Au']:
metal_name = 'Au'
x = metal(metal_name)

n_GCNs = 100

if metal_name == 'Pt':
    GCN_vec = np.linspace(6,10,n_GCNs)
elif metal_name == 'Au':
    GCN_vec = np.linspace(2,8,n_GCNs)

det_vol = np.zeros(n_GCNs)
UQ_uncorr_vol = np.zeros(n_GCNs)
UQ_corr_vol = np.zeros(n_GCNs)
UQ_corr_vol_upper = np.zeros(n_GCNs)        # 95th percentile
UQ_corr_vol_lower = np.zeros(n_GCNs)        # 5th percentile

    
for i in range(n_GCNs):

    print 'GCN: ' + str(GCN_vec[i])
    
    # Data for deterministic volcano
    BEs = x.get_BEs(GCN_vec[i], uncertainty = False)
    rate = ORR_rate(BEs[0], BEs[1],explicit=False)
    det_vol[i] = rate
    
    n_MC_samples = 10000
    #n_MC_samples = 100     # for fast testing
    upper_ind = int(n_MC_samples * 0.95)
    lower_ind = int(n_MC_samples * 0.05)
    data_uncorr = np.zeros(n_MC_samples)
    data_corr = np.zeros(n_MC_samples)
    
    for j in range(n_MC_samples):

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
    UQ_corr_vol_upper[i] = data_corr[upper_ind]
    UQ_corr_vol_lower[i] = data_corr[lower_ind]
    
'''
smooth the data
'''
y = np.log(UQ_corr_vol_upper)
yhat = savitzky_golay(y, 9, 3) # window size 51, polynomial order 3
UQ_corr_vol_upper = np.exp(yhat)

y = np.log(UQ_corr_vol_lower)
yhat = savitzky_golay(y, 9, 3) # window size 51, polynomial order 3
UQ_corr_vol_lower = np.exp(yhat)

y = np.log(UQ_uncorr_vol)
yhat = savitzky_golay(y, 9, 3) # window size 51, polynomial order 3
UQ_uncorr_vol = np.exp(yhat)

y = np.log(UQ_corr_vol)
yhat = savitzky_golay(y, 9, 3) # window size 51, polynomial order 3
UQ_corr_vol = np.exp(yhat)
    
'''
Normalize the data - should not be needed
'''

#Pt111_norm = 1.35703847925e-15      # experimental value in mA / atom for Pt(111) at GCN = 7.5    
#Pt111_now = np.exp( np.interp( 7.5, GCN_vec, np.log(det_vol) ) )
#norm_factor = Pt111_norm / Pt111_now
#print Pt111_norm
#print Pt111_now
#print norm_factor

    
'''
Write data to numpy files
'''

all_data = np.vstack([GCN_vec, det_vol, UQ_uncorr_vol, UQ_corr_vol, UQ_corr_vol_upper, UQ_corr_vol_lower])
#all_data[1::,:] = all_data[1::,:] * norm_factor
np.save(metal_name+'_UQ_vol.npy', all_data)   
    
