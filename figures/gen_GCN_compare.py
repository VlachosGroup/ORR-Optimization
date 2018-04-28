'''
Generates data comparing activities of different GCNs
'''

import os
import sys
this_folder = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(this_folder,'..','structure')) 

from metal import metal
from ORR import ORR_rate
import numpy as np
from scipy.stats import norm

from multiprocessing import Pool


metal_name = 'Au'

# Separate function for use in parallelization
def square(z):
    #for metal_name in ['Pt','Au']:
    
    x = metal(metal_name)

    if metal_name == 'Pt':
        GCN_vec = np.array([z,8.29])
    elif metal_name == 'Au':
        GCN_vec = np.array([z,5.75])
        
    n_GCNs = len(GCN_vec)
    n_MC_samples = 100000    
    all_samples = np.zeros([n_GCNs, n_MC_samples])

        
    for i in xrange(n_GCNs):

        print 'GCN: ' + str(GCN_vec[i])
        
        
        #n_MC_samples = 100     # for fast testing
        upper_ind = int(n_MC_samples * 0.95)
        lower_ind = int(n_MC_samples * 0.05)
        data_uncorr = np.zeros(n_MC_samples)
        data_corr = np.zeros(n_MC_samples)
        
        for j in xrange(n_MC_samples):
        
            # Correlated data
            BEs = x.get_BEs(GCN_vec[i], uncertainty = True, correlations = True)
            rate = ORR_rate(BEs[0], BEs[1],explicit=False)
            data_corr[j] = rate
        
        data_corr = np.sort(data_corr)
        all_samples[i,:] = data_corr

    pdf1 = all_samples[0,:]
    pdf2 = all_samples[1,:]     # The GCN of comparison
    total = 0
    for j in xrange(n_MC_samples):
        total += float(np.sum( pdf2 < pdf1[j] )) / n_MC_samples
        
    return total / n_MC_samples

if __name__ == '__main__':                 # Need this line to make parallelization work

    n_procs = 16
    if metal_name == 'Pt':
        GCN_min = 6.0
        GCN_max = 10.0
    elif metal_name == 'Au':
        GCN_min = 3.0
        GCN_max = 8.0
    
    n_points = 100
    x_vec_npy = np.linspace(GCN_min, GCN_max, n_points)
    x_vec = [x_vec_npy[i] for i in xrange(n_points)]
    
    # Run in parallel
    pool = Pool(processes = n_procs)
    y_vec = pool.map(square, x_vec)
    pool.close()
    
    np.save(metal_name + '_compare_data', np.vstack([x_vec,y_vec]))