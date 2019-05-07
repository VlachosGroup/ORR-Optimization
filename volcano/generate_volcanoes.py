# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 12:27:51 2018

@author: lansf
"""
from __future__ import division
import os
import sys
import numpy as np
from orr_optimizer.orr_mkm import ORR_MKM
import matplotlib.pyplot as plt
from orr_optimizer.metal import metal
from orr_optimizer.ORR import ORR_rate
metal_name = 'Pt'
x = metal(metal_name)

GCN_terrace = np.linspace(4,9,50)
MKM = ORR_MKM('terrace')
rate_terrace = []
coverage_terrace = []
rate_0cov = []
rate_impl = []
rate_vallejo = []
for GCN in GCN_terrace:
    t, solution = MKM.get_coverage(GCN)
    coverage_terrace.append(solution[-1])
    rate = MKM.get_rate(GCN,solution[-1])
    """explicit rate with coverage effects"""
    rate_terrace.append(rate)
    BEs = x.get_BEs(GCN, uncertainty = False)
    """rate from just vallejo's data"""
    rate_vallejo.append(ORR_rate(BEs[0], BEs[1],explicit=False))
    """rate with explicit solvation effects"""
    rate_0cov.append(ORR_rate(BEs[0], BEs[1],explicit=True))
    """rate with implicit solvation and coverage effects"""
    """(implicit + coverage effect) = implicit + (explicit + coverage) - (explicit at 0 coverage)"""
    rate_impl.append(ORR_rate(BEs[0] + MKM.dGdOH(solution[-1,0:3],MKM.popt,0) - MKM.dGdOH(np.zeros(3),MKM.popt,0)
    , BEs[1] + MKM.dGdOOH(solution[-1,0:3],MKM.popt,0) - MKM.dGdOOH(np.zeros(3),MKM.popt,0),explicit=False))

MKM = ORR_MKM('edge')
rate_edge = []
coverage_edge = []
for GCN in GCN_terrace:
    t, solution = MKM.get_coverage(GCN)
    coverage_edge.append(solution[-1])
    rate = MKM.get_rate(GCN,solution[-1])
    rate_edge.append(rate)

MKM = ORR_MKM('cavity_edge')
rate_cavity = []
coverage_cavity = []
for GCN in GCN_terrace:
    t, solution = MKM.get_coverage([GCN,5.167+GCN-8.5])
    coverage_cavity.append(solution[-1])
    rate = MKM.get_rate([GCN,5.167+GCN-8.5],solution[-1])
    rate_cavity.append(rate[0])
coverage_cavity = np.array(coverage_cavity)

MKM = ORR_MKM('cavity_edge')
rate_cedge = []
coverage_cedge = []
for GCN in GCN_terrace:
    t, solution = MKM.get_coverage([8.5+GCN-5.167,GCN])
    coverage_cedge.append(solution[-1])
    rate = MKM.get_rate([8.5+GCN-5.167,GCN],solution[-1])
    rate_cedge.append(rate[1])
coverage_cedge = np.array(coverage_cedge)

"""terrace coverage"""
plt.figure(0)
plt.plot(GCN_terrace,coverage_terrace)
plt.legend(['OH','OOH','O (fcc)','O (atop)'])
plt.ylabel('coverage [ML]')
plt.xlabel('GCN')

"""GCN 6.4 edge coverage"""
plt.figure(1)
plt.plot(GCN_terrace,coverage_edge)
plt.legend(['OH','OOH','O (fcc)','O (atop)'])
plt.ylabel('coverage [ML]')
plt.xlabel('GCN')
plt.savefig('Josh1.png', format='png')
plt.close()

"""GCN 5.1 edge coverage"""
plt.figure(2)
plt.plot(GCN_terrace,coverage_cedge[:,0],GCN_terrace,coverage_cedge[:,2],GCN_terrace,coverage_cedge[:,4],GCN_terrace,coverage_cedge[:,5])
plt.legend(['OH','OOH','O (fcc)','O (atop)'])
plt.ylabel('coverage [ML]')
plt.xlabel('GCN')
plt.savefig('Josh2.png', format='png')
plt.close()

"""GCN 8.5 cavity coverage"""
plt.figure(3)
plt.plot(GCN_terrace,coverage_cavity[:,1],GCN_terrace,coverage_cavity[:,3],GCN_terrace,coverage_cavity[:,6],GCN_terrace,coverage_cavity[:,7])
plt.legend(['OH','OOH','O (fcc)','O (atop)'])
plt.ylabel('coverage [ML]')
plt.xlabel('GCN')
plt.savefig('Josh3.png', format='png')
plt.close()

"""Generating 2D map of rate for cavity and edge sites"""
'''
Normalize
'''

rate_terrace = np.array(rate_terrace)
rate_vallejo = np.array(rate_vallejo)
rate_0cov = np.array(rate_0cov)
rate_impl = np.array(rate_impl)
rate_edge = np.array(rate_edge)
rate_cavity = np.array(rate_cavity)
rate_cedge = np.array(rate_cedge)
#rate_cavity_edge = np.array(rate_cavity_edge)

Pt111_norm = 1.35703847925e-15      # experimental value in mA / atom for Pt(111) at GCN = 7.5
Pt111_now = np.exp( np.interp( 7.5, GCN_terrace, np.log(rate_terrace) ) )
Pt111_0cov = np.exp( np.interp( 7.5, GCN_terrace, np.log(rate_0cov) ) )
Pt111_impl = np.exp( np.interp( 7.5, GCN_terrace, np.log(rate_impl) ) )

rate_terrace = rate_terrace * Pt111_norm / Pt111_now
rate_0cov = rate_0cov * Pt111_norm / Pt111_0cov
rate_impl = rate_impl * Pt111_norm / Pt111_impl
rate_edge = rate_edge * Pt111_norm / Pt111_now
rate_cavity = rate_cavity * Pt111_norm/ Pt111_now
rate_cedge = rate_cedge * Pt111_norm / Pt111_now

print (np.exp( np.interp( 7.5, GCN_terrace, np.log(rate_terrace) ) )) # should be Pt111_norm

'''
Plot volcano for different edges, cavity and terrace based Hamiltonians
'''

plt.figure(4)
plt.plot(GCN_terrace,np.log(rate_terrace),GCN_terrace,np.log(rate_edge),GCN_terrace,np.log(rate_cedge),GCN_terrace,np.log(rate_cavity))
plt.legend(['Terrace (GCN=7.5)','Edge (GCN=6.4)','Edge (GCN = 5.1)','Cavity (GCN = 8.5)'],loc=4,frameon=False)
plt.ylabel('log(rate) log[mA/atom]')
plt.xlabel('GCN')
plt.savefig('Josh4.png', format='png')
plt.close()

"""plotting 2D volcano"""
'''
rate_list2 = np.array([i if i>0 else 10**16 for i in rate_cavity_edge])
rate_list2 = np.array([i if i<> 10**16 else min(rate_list2) for i in rate_list2])*Pt111_norm / Pt111_now
rate_matrix = (rate_list2).reshape(50,50)
GCN_cavity_edge = np.array(GCN_cavity_edge)
plt.figure(4)
#plt.pcolor(np.linspace(4,8,50), np.linspace(6,10,50), rate_matrix,cmap='jet')
plt.scatter(GCN_cavity_edge[:,1],GCN_cavity_edge[:,0],c=np.log10(rate_list2),cmap='jet',edgecolors='face')
plt.ylabel('cavity GCN',size=20)
plt.xlabel('edge GCN',size=20)
plt.xticks(size=18)
plt.yticks(size=18)
plt.colorbar(label='log$_{10}$(rate) log$_{10}$[mA/atom]')
plt.gcf().subplots_adjust(bottom=0.15)
plt.gcf().subplots_adjust(left=0.17)
plt.show()
'''

"""comparing Pt111 rates"""
plt.figure(5)
plt.plot(GCN_terrace,np.log(rate_vallejo),zorder=10)
plt.plot(GCN_terrace,np.log(rate_0cov),GCN_terrace,np.log(rate_impl),GCN_terrace,np.log(rate_terrace))
plt.legend(['impl solv @ zero cov','expl solv @ zero cov','impl solv @ SS cov','expl solv @ SS cov'],loc=4,frameon=False)
plt.ylabel('log(rate) log[mA/atom]')
plt.xlabel('GCN')
plt.savefig('Josh5.png', format='png')
plt.close()

plt.figure(6)
plt.plot(GCN_terrace, rate_terrace, marker=None, linestyle = '-', color = 'k', label = 'terrace')
plt.plot(GCN_terrace, rate_edge, marker=None, linestyle = '-', color = 'r', label = 'edge')
plt.plot(GCN_terrace, rate_cedge, marker=None, linestyle = '-', color = 'b', label = 'cedge')
plt.plot(GCN_terrace, rate_cavity, marker=None, linestyle = '-', color = 'g', label = 'cavity')
plt.plot(GCN_terrace, rate_vallejo, marker=None, linestyle = '-', color = 'm', label = 'Calle-Vallejo')
plt.legend(loc=4, prop={'size':20}, frameon=False)
plt.ylabel('rate [mA/atom]')
plt.xlabel('GCN')
plt.yscale('log')
plt.savefig('Marcel.png', format='png')
plt.close()

# Save volcano data in units of mA/atom
data_out = np.transpose( np.vstack([GCN_terrace, rate_terrace, rate_edge, rate_cedge, rate_cavity, rate_vallejo]) )
np.save('all_volcanos.npy', data_out)
