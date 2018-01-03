# -*- coding: utf-8 -*-
"""
Regresses surface energy as a function of coverage. The derivative is taken 
to determine adsorption energies at different coverages and the coverage at
steady state is calculated using an MKM.

@author: lansf
"""
from __future__ import division
import os
from pandas import read_csv
import numpy as np
from scipy.optimize import curve_fit
from matplotlib import rcParams
from scipy.integrate import odeint
from SteadyState import ORR_rate
rcParams['legend.numpoints'] = 1
data_fldr = 'Surface_Energies.csv'
data_fldr = os.path.expanduser(data_fldr)
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
s,tp,u,x,y,z,a,b,c = popt
def dGdOH(Coverageinput,s,tp,u,x,y,z,a):
    OHcov, OOHcov, Ocov = Coverageinput
    dGval = a + u*s*(tp*Ocov+OHcov)**(u-1) + z*x*(y*Ocov+OHcov)**(z-1)*OOHcov
    return dGval
    
def dGdO(Coverageinput,s,tp,u,x,y,z,b):
    OHcov, OOHcov, Ocov = Coverageinput
    dGval = b + tp*u*s*(tp*Ocov+OHcov)**(u-1)+y*z*x*(y*Ocov+OHcov)**(z-1)*OOHcov
    return dGval
    
def dGdOOH(Coverageinput,x,y,z,c):
    OHcov, OOHcov, Ocov = Coverageinput
    dGval = c + x*(y*Ocov+OHcov)**z
    return dGval

"""zero coverage adsorption energies"""
dGdOHzero = dGdOH(np.zeros(3),s,tp,u,x,y,z,a)
print(dGdOHzero)
dGdOOHzero = dGdOOH(np.zeros(3),x,y,z,c)
print(dGdOOHzero)

t = np.linspace(0, 10**-4, 200)
sol = odeint(ORR_rate, [0,0,0,0], t, args=(dGdOH,dGdOOH,dGdO,s,tp,u,x,y,z,a,b,c,0.9))
"""printing coverages"""
print('OH',sol[-1,0],'OOH',sol[-1,1],'O (fcc)',sol[-1,2])

"""printing adsorption energies at coverage"""
dGdOHcov = dGdOH(np.array([sol[-1,0],sol[-1,1],sol[-1,2]]),s,tp,u,x,y,z,a)
print(dGdOHcov)
dGdOOHcov = dGdOOH(np.array([sol[-1,0],sol[-1,1],sol[-1,2]]),x,y,z,c)
print(dGdOOHcov)

"""printing coverage effect"""
print('OH',dGdOHcov-dGdOHzero,'OOH',dGdOOHcov-dGdOOHzero)