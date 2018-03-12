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


    '''
    Plot PDF of GCN relation errors
    '''

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

    fig = plt.figure()
    plt.contourf(BE_OH_err_vec, BE_OOH_err_vec, rel_prob)
    plt.colorbar()  
    #points = plt.plot(x.res_OH, x.res_OOH, 'o', markeredgecolor='k', markerfacecolor=[0.5, 0.5, 0.5])
    points = plt.plot(x.res_OH, x.res_OOH, 'o', markeredgecolor='k', markeredgewidth=2, markerfacecolor='None')
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.xlabel(r'OH* binding energy error, $e_{OH}$ (eV)',size=20)
    plt.ylabel(r'OOH* binding energy error, $e_{OOH}$ (eV)',size=20)
    #plt.zlabel('Relative Prob.',size=24)
    plt.tight_layout()
    plt.savefig(metal_name + '_heat_map.png', format='png', dpi=600)
    plt.close()