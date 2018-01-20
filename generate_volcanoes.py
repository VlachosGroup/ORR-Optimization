# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 12:27:51 2018

@author: lansf
"""
import numpy as np
from orr_mkm import ORR_MKM
import matplotlib.pyplot as plt

MKM = ORR_MKM('terrace')
GCN_terrace = np.linspace(2,9,100)
rate_terrace = []
for GCN in GCN_terrace:
    t, solution = MKM.get_coverage(GCN)
    rate = MKM.get_rate(GCN,solution[-1])
    rate_terrace.append(rate)

MKM = ORR_MKM('cavity_edge')
GCN_cavity = np.linspace(2,9,100)
rate_cavity = []
for GCN in GCN_cavity:
    t, solution = MKM.get_coverage([GCN,5.167])
    rate = MKM.get_rate([GCN,5.167],solution[-1])
    rate_cavity.append(rate[0])

MKM = ORR_MKM('cavity_edge')
GCN_edge = np.linspace(2,9,100)
rate_edge = []
for GCN in GCN_edge:
    t, solution = MKM.get_coverage([8.5,GCN])
    rate = MKM.get_rate([8.5,GCN],solution[-1])
    rate_edge.append(rate[1])
    
plt.plot(GCN_terrace,np.log10(rate_terrace),GCN_cavity,np.log10(rate_cavity),GCN_edge,np.log10(rate_edge))
plt.legend(['Terrace','Cavity','Edge'])
plt.ylabel('log10(rate)')
plt.xlabel('GCN')