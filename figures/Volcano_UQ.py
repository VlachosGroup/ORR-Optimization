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

for metal_name in ['Pt','Au']:

    x = metal(metal_name)


    n_mesh = 100
    BE_OH_err_vec = np.linspace(-0.3,0.3,n_mesh)
    BE_OOH_err_vec = np.linspace(-0.3,0.3,n_mesh)

    BE_OH_err_mesh, BE_OOH_err_mesh = np.meshgrid(BE_OH_err_vec, 
        BE_OOH_err_vec, sparse=False, indexing='xy')
    rel_prob = np.zeros(BE_OH_err_mesh.shape)
    for i in range(n_mesh):
        for j in range(n_mesh):
            pcas = np.matmul(np.array([BE_OH_err_mesh[i,j], BE_OOH_err_mesh[i,j]]), x.pca_inv )
            rel_prob[i,j] = norm.pdf(pcas[0] / x.sigma_pca_1 ) * norm.pdf(pcas[1] / x.sigma_pca_2 )

    mat.rcParams['mathtext.default'] = 'regular'
    mat.rcParams['text.latex.unicode'] = 'False'
    mat.rcParams['legend.numpoints'] = 1
    mat.rcParams['lines.linewidth'] = 4
    mat.rcParams['lines.markersize'] = 12



    '''
    Plot volcano plots
    '''

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
        
    # Read in volcano data 
        
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
    plt.savefig(metal_name + '_volcano.png', format='png', dpi=600)
    plt.close()