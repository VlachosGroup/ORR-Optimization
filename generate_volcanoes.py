# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 12:27:51 2018

@author: lansf
"""
from __future__ import division
import numpy as np
from orr_mkm import ORR_MKM
import matplotlib.pyplot as plt

MKM = ORR_MKM('terrace')
GCN_terrace = np.linspace(2,9,100)
rate_terrace = []
coverage_terrace = []
for GCN in GCN_terrace:
    t, solution = MKM.get_coverage(GCN)
    coverage_terrace.append(solution[-1])
    rate = MKM.get_rate(GCN,solution[-1])
    rate_terrace.append(rate)

MKM = ORR_MKM('cavity_edge')
GCN_cavity = np.linspace(2,9,100)
rate_cavity = []
coverage_cavity = []
for GCN in GCN_cavity:
    t, solution = MKM.get_coverage([GCN,5.167])
    coverage_cavity.append(solution[-1])
    rate = MKM.get_rate([GCN,5.167],solution[-1])
    rate_cavity.append(rate[0])
coverage_cavity = np.array(coverage_cavity)
MKM = ORR_MKM('cavity_edge')
GCN_edge = np.linspace(2,9,100)
rate_edge = []
coverage_edge = []
for GCN in GCN_edge:
    t, solution = MKM.get_coverage([8.5,GCN])
    coverage_edge.append(solution[-1])
    rate = MKM.get_rate([8.5,GCN],solution[-1])
    rate_edge.append(rate[1])
coverage_edge = np.array(coverage_edge)

plt.figure(0)
plt.plot(GCN_terrace,coverage_terrace)
plt.legend(['OH','OOH','O (fcc)','O (atop)'])
plt.ylabel('coverage [ML]')
plt.xlabel('GCN')

plt.figure(1)
plt.plot(GCN_cavity,coverage_cavity[:,1],GCN_cavity,coverage_cavity[:,3]
,GCN_cavity,coverage_cavity[:,4],GCN_cavity,coverage_cavity[:,5])
plt.legend(['OH','OOH','O (fcc)','O (atop)'])
plt.ylabel('coverage [ML]')
plt.xlabel('GCN')

plt.figure(2)
plt.plot(GCN_edge,coverage_edge[:,0],GCN_edge,coverage_edge[:,2]
,GCN_edge,coverage_edge[:,4],GCN_edge,coverage_edge[:,5])
plt.legend(['OH','OOH','O (fcc)','O (atop)'])
plt.ylabel('coverage [ML]')
plt.xlabel('GCN')

'''
Normalize
'''

rate_terrace = np.array(rate_terrace)
rate_cavity = np.array(rate_cavity)
rate_edge = np.array(rate_edge)
    
Pt111_norm = 1.35703847925e-15      # experimental value in mA / atom for Pt(111) at GCN = 7.5
Pt111_now = np.exp( np.interp( 7.5, GCN_terrace, np.log(rate_terrace) ) )

rate_terrace = rate_terrace * Pt111_norm / Pt111_now
rate_cavity = rate_cavity * Pt111_norm / Pt111_now
rate_edge = rate_edge * Pt111_norm / Pt111_now

print np.exp( np.interp( 7.5, GCN_terrace, np.log(rate_terrace) ) )
raise NameError('stop')

'''
Plot volcano
'''
plt.figure(3)
plt.plot(GCN_terrace,np.log10(rate_terrace),GCN_cavity,np.log10(rate_cavity),GCN_edge,np.log10(rate_edge))
plt.legend(['Terrace','Cavity','Edge'])
plt.ylabel('log$_{10}$(rate) [mA/atom]')
plt.xlabel('GCN')


# Output into data table

vol_data = np.transpose( np.vstack([GCN_terrace, rate_terrace, rate_cavity, rate_edge]) )
np.save('volcano_data.npy', vol_data)