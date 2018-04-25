# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 12:27:51 2018

@author: lansf
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mat

all_vols = np.load('../structure/all_volcanos.npy')

Na = 6.022e23
mA_to_kA = 1.0e-6

# Convert from [miliAmps per atom] to [kiloAmps per mol]
all_vols[:,1::] = all_vols[:,1::] * mA_to_kA * Na

GCN_terrace = all_vols[:,0]
rate_terrace = all_vols[:,1]
rate_edge = all_vols[:,2]
rate_cedge = all_vols[:,3]
rate_cavity = all_vols[:,4]
rate_vallejo = all_vols[:,5]

# GCNs used to make plots
terrace_GCN = 7.5
edge_GCN = 5.1
cedge_GCN = 6.4
cavity_GCN = 8.5

terrace_act = np.exp( np.interp( terrace_GCN, GCN_terrace, np.log(rate_terrace) ) )
#edge_act = np.exp( np.interp( cedge_GCN, GCN_terrace, np.log(rate_edge) ) )
edge_act = np.interp( cedge_GCN, GCN_terrace, rate_edge) 
cedge_act = np.exp( np.interp( edge_GCN, GCN_terrace, np.log(rate_cedge) ) )
cavity_act = np.exp( np.interp( cavity_GCN, GCN_terrace, np.log(rate_cavity) ) )

mat.rcParams['mathtext.default'] = 'regular'
mat.rcParams['text.latex.unicode'] = 'False'
mat.rcParams['legend.numpoints'] = 1
mat.rcParams['lines.linewidth'] = 2
mat.rcParams['lines.markersize'] = 12

plt.figure(6)
plt.plot(GCN_terrace, rate_terrace, marker=None, linestyle = '-', color = 'k', label = 'terrace')
plt.plot(np.array([terrace_GCN]), np.array([terrace_act]), marker='o', linestyle = '-', color = 'k', label = None)
plt.plot(GCN_terrace, rate_edge, marker=None, linestyle = '-', color = 'r', label = 'edge')
plt.plot([edge_GCN], [edge_act], marker='o', linestyle = '-', color = 'r', label = None)
plt.plot(GCN_terrace, rate_cedge, marker=None, linestyle = '-', color = 'b', label = 'cedge')
plt.plot([cedge_GCN], [cedge_act], marker='o', linestyle = '-', color = 'b', label = None)
plt.plot(GCN_terrace, rate_cavity, marker=None, linestyle = '-', color = 'g', label = 'cavity')
plt.plot([cavity_GCN], [cavity_act], marker='o', linestyle = '-', color = 'g', label = None)
plt.plot(GCN_terrace, rate_vallejo, marker=None, linestyle = '-', color = 'm', label = 'Calle-Vallejo')
plt.legend(loc=4, prop={'size':20}, frameon=False)
plt.xlabel('Generalized coordination number', size=24)
plt.ylabel(r'Current, $i$ [kA/mol]', size=24)

plt.yscale('log')
plt.xticks(size=24)
plt.yticks(size=24)
plt.tight_layout()

plt.savefig('JL_vols.png', format='png', dpi = 600)
plt.close()
