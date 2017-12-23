'''
Uses uncertainty in the GCN fits to plot a volcano with uncertainty
'''

from metal import metal
from ORR import ORR_rate
import numpy as np
from scipy.stats import norm

import matplotlib.pyplot as plt
import matplotlib as mat

metal_name = 'Pt'
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
plt.xlabel('BE_OH error (eV)',size=20)
plt.ylabel('BE_OOH error (eV)',size=20)
#plt.zlabel('Relative Prob.',size=24)
plt.tight_layout()
plt.savefig(metal_name + '_heat_map.png', format='png', dpi=600)
plt.close()


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
    
for i in xrange(n_GCNs):

    print 'GCN: ' + str(GCN_vec[i])
    
    # Data for deterministic volcano
    BEs = x.get_BEs(GCN_vec[i], uncertainty = False)
    rate = ORR_rate(BEs[0], BEs[1],explicit=True,coverage=True)
    det_vol[i] = rate
    
    n_MC_samples = 1000
    data_uncorr = np.zeros(n_MC_samples)
    data_corr = np.zeros(n_MC_samples)
    
    for j in xrange(n_MC_samples):

        # Unocorrelated data
        BEs = x.get_BEs(GCN_vec[i], uncertainty = True, correlations = False)
        rate = ORR_rate(BEs[0], BEs[1],explicit=True,coverage=True)
        data_uncorr[j] = rate
    
        # Correlated data
        BEs = x.get_BEs(GCN_vec[i], uncertainty = True, correlations = True)
        rate = ORR_rate(BEs[0], BEs[1],explicit=True,coverage=True)
        data_corr[j] = rate
    
    data_uncorr = np.sort(data_uncorr)
    data_corr = np.sort(data_corr)
    
    UQ_uncorr_vol[i] = np.exp( np.mean( np.log(data_uncorr)))
    UQ_corr_vol[i] = np.exp( np.mean( np.log(data_corr)))
    
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