# -*- coding: utf-8 -*-
"""
Regresses surface energy as a function of coverage for the Pt surface
that contains a cavity with edge atoms. The derivative is taken to determine
adsorption energies at different coverages and the coverage at steady state is 
calculated using an MKM.
@author: lansf
"""
from __future__ import division
import os
from pandas import read_csv
import numpy as np
from scipy.optimize import curve_fit
from matplotlib import rcParams
from scipy.integrate import odeint
from SteadyState_cavity_edge import ORR_rate
from metal import metal
metal_name = 'Pt'
xBE = metal(metal_name)
rcParams['legend.numpoints'] = 1
data_fldr = 'Surface_Energies.csv'
data_fldr = os.path.expanduser(data_fldr)
data_cavity = 'Surface_Energies_cavity.csv'
data_cavity = os.path.expanduser(data_cavity)

E_H2Og = -14.219432

#Experimental value of H2O solvation from gas to liquid from:
#G. Schüürmann, M. Cossi, V. Barone, and J. Tomasi, The Journal of Physical Chemistry A 102, 6706 (1998).
#Experimental H2O solvation value confirmed by
#M. D. Liptak and G. C. Shields, J. Am. Chem. Soc. 123, 7314 (2001). and
#M. W. Palascak and G. C. Shields, The Journal of Physical Chemistry A 108, 3692 (2004).
Esolv_H2O = -0.2736
E_solv = [-0.575, -0.480] #OH* and OOH*
E_g = [-7.73, -13.25, -1.90] #OH and OOH and O
Pt7_5 = [-208.21404,-218.12374,-222.52624] #P111 OH and OOH Energies without water
CovDat = read_csv(data_fldr)
CovDatCav = read_csv(data_cavity)
E7H2O = -379.78779 # water in cavity
E6H2O = -365.04325 # removing H2O from cavity
Esolv_H2O_explicit = E7H2O-E6H2O-E_H2Og #this is solvation energy of H2O interacting with a surface
G_H2Ol = E_H2Og + Esolv_H2O_explicit
Coverages = np.array([CovDat.OHcov,CovDat.Ocov,CovDat.OOHcov])
WaterReplacement = np.sum(CovDat[['OHcov','OOHcov']],axis=1)*9*G_H2Ol

def Gsurf(Coverageinput,s,tp,u,x,y,z,a,b,c):
    OHcov, Ocov, OOHcov = Coverageinput
    Gval = a*OHcov + b*Ocov + c*OOHcov + s*(tp*Ocov+OHcov)**u + x*(y*Ocov+OHcov)**z*OOHcov
    return Gval

Go = -385.40342
Energies = CovDat.Energy.as_matrix() + WaterReplacement - Go
lmin = 0
lmax = 30
emin = 1
emax=4
popt, pcov = curve_fit(Gsurf,Coverages,Energies/9.0,bounds=(np.array([lmin,lmin,emin,lmin,lmin,emin,-20,-20,-20]),np.array([lmax,lmax,emax,lmax,lmax,emax,0,0,0])))

sO,tpO,uO,xO,yO,zO,aO,bO,cO = popt
    
def dGdO(Coverageinput,s,tp,u,x,y,z,b):
    OHcov, Ocov, OOHcov = Coverageinput
    dGval = b + tp*u*s*(tp*Ocov+OHcov)**(u-1)+y*z*x*(y*Ocov+OHcov)**(z-1)*OOHcov
    return dGval

Coverages = np.array([CovDatCav.OH_edge,CovDatCav.OH_cavity,CovDatCav.OOH_edge,CovDatCav.OOH_cavity])
WaterReplacement = np.sum(Coverages,axis=0)*9*G_H2Ol
def GsurfCav(Coverageinput,x,x2,x3,y,z,a,b,c,d):
    OHedge, OHcav, OOHedge, OOHcav = Coverageinput
    Gval = a*OHedge + b*OHcav + c*OOHedge + d*OOHcav + x*(y*OHedge+OOHedge)**z + x2*(OHedge+OOHedge)*OHcav + x3*(OHedge+OOHedge)*OOHcav
    return Gval

Go = -365.04325
Energies = CovDatCav.Energy.as_matrix() + WaterReplacement - Go
lmin = 0
lmax = 30
emin = 1
emax=4
popt, pcov = curve_fit(GsurfCav,Coverages,Energies/9.0,bounds=(np.array([lmin,lmin,lmin,lmin,emin,-20,-20,-20,-20]),np.array([lmax,lmax,lmax,lmax,emax,0,0,0,0])))
x,x2,x3,y,z,a,b,c,d = popt

def dGdOHedge(Coverageinput,x,x2,x3,y,z,a):
    OHedge, OHcav, OOHedge, OOHcav, Oedge = Coverageinput
    dGval = a + y*x*z*(y*OHedge+OOHedge+y*Oedge)**(z-1) + x2*OHcav + x3*OOHcav
    return dGval

def dGdOHcav(Coverageinput,b,x2):
    OHedge, OHcav, OOHedge, OOHcav, Oedge = Coverageinput
    dGval = b + x2*(OHedge+OOHedge+Oedge)
    return dGval

def dGdOOHedge(Coverageinput,x,x2,x3,y,z,c):
    OHedge, OHcav, OOHedge, OOHcav, Oedge = Coverageinput
    dGval = c + x*z*(y*OHedge+OOHedge+y*Oedge)**(z-1) + x2*OHcav + x3*OOHcav
    return dGval

def dGdOOHcav(Coverageinput,d,x3):
    OHedge, OHcav, OOHedge, OOHcav,Oedge = Coverageinput
    dGval = d + x3*(OHedge + OOHedge+Oedge)
    return dGval
 
t = np.linspace(0, 20, 100)
sol = odeint(ORR_rate, [0,0,0,0,0,0], t, args=(dGdOHedge,dGdOHcav,dGdOOHedge,dGdOOHcav,dGdO,sO,tpO,uO,xO,yO,zO,aO,bO,cO,x,x2,x3,y,z,a,b,c,d,0.9))

"""printing coverages"""
print('OHedge',sol[-1,0],'OHcav',sol[-1,1],'OOHedge',sol[-1,2],'OOHcav',sol[-1,3],'O (fcc)',sol[-1,4],'O (atop)',sol[-1,5])

"""printing adsorption energies at coverage"""
dGdOHedgecov = dGdOHedge(np.array([sol[-1,0],sol[-1,1],sol[-1,2],sol[-1,3],sol[-1,4]]),x,x2,x3,y,z,a)
dGdOHcavcov = dGdOHcav(np.array([sol[-1,0],sol[-1,1],sol[-1,2],sol[-1,3],sol[-1,4]]),b,x2)
dGdOOHedgecov = dGdOOHedge(np.array([sol[-1,0],sol[-1,1],sol[-1,2],sol[-1,3],sol[-1,4]]),x,x2,x3,y,z,c)
dGdOOHcavcov = dGdOOHcav(np.array([sol[-1,0],sol[-1,1],sol[-1,2],sol[-1,3],sol[-1,4]]),d,x3)
print(dGdOHedgecov)
print(dGdOHcavcov)
print(dGdOOHedgecov)
print(dGdOOHcavcov)
"""printing coverage effect"""
dGdOHedgezero = dGdOHedge(np.zeros(5),x,x2,x3,y,z,a)
dGdOHcavzero = dGdOHcav(np.zeros(5),b,x2)
dGdOOHedgezero = dGdOOHedge(np.zeros(5),x,x2,x3,y,z,c)
dGdOOHcavzero = dGdOOHcav(np.zeros(5),d,x3)

print(dGdOHedgecov-dGdOHedgezero)
print(dGdOHcavcov-dGdOHcavzero)
print(dGdOOHedgecov-dGdOOHedgezero)
print(dGdOOHcavcov-dGdOOHcavzero)

#scaling coverage effect based on implicit BEs of OH at low coverage
BEsOH = xBE.get_BEs(np.array([5.167,7.5,8.5]), uncertainty = False)[0]
OHcoveffect = np.array([0.4253,0.8998,0])
OOHcoveffect = np.array([0.5493,0.2497,0.1205])
OHlowGCN = np.polyfit(BEsOH[:2], OHcoveffect[:2], 1)
print(OHlowGCN)
OHhighGCN = np.polyfit(BEsOH[1:], OHcoveffect[1:], 1)
print(OHhighGCN)
OOHlowGCN = np.polyfit(BEsOH[:2], OOHcoveffect[:2], 1)
print(OOHlowGCN)
OOHhighGCN = np.polyfit(BEsOH[1:], OOHcoveffect[1:], 1)
print(OOHhighGCN)

