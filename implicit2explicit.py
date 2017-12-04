# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 22:36:33 2017

@author: lansf
"""
import os
from pandas import read_csv
import numpy as np
import matplotlib.pyplot as plt
from metal import metal
data_fldr = '~/Box Sync/Synced_Files/Coding/Research/ORR Marcel/ORR_Honeycomb_Energies.csv'
data_fldr = os.path.expanduser(data_fldr)
E_H2Og = -14.219432

#Experimental value of H2O solvation from gas to liquid from:
#G. Schüürmann, M. Cossi, V. Barone, and J. Tomasi, The Journal of Physical Chemistry A 102, 6706 (1998).
#Experimental H2O solvation value confirmed by
#M. D. Liptak and G. C. Shields, J. Am. Chem. Soc. 123, 7314 (2001). and
#M. W. Palascak and G. C. Shields, The Journal of Physical Chemistry A 108, 3692 (2004).
Esolv_H2O = -0.2736
G_H2Ol = E_H2Og + Esolv_H2O
E_solv = [-0.575, -0.480] #OH* and OOH*
E_g = [-7.53, -13.26] #OH and OOH
Pt7_5 = [-208.21404,-218.12374,-222.52624] #P111 OH and OOH Energies without water
CovDat = read_csv(data_fldr)

#conversion factor from implicit to explicit adsorption energy
EOHimpl7_5 = Pt7_5[1] - Pt7_5[0] - E_g[0] + E_solv[0]
EOOHimpl7_5 = Pt7_5[2] - Pt7_5[0] - E_g[1] + E_solv[1]
EOHexpl7_5 = CovDat['Pt111+OH']-CovDat['Pt111']-E_g[0]+G_H2Ol
EOOHexpl7_5 = CovDat['Pt111+OOH']-CovDat['Pt111']-E_g[1]+G_H2Ol
EOHimpl2expl = EOHexpl7_5[0]-EOHimpl7_5
EOOHimpl2expl = EOOHexpl7_5[0]-EOOHimpl7_5

#conversion factor to account for the presence of oxygen
EOHexpl7_5O = CovDat['Pt111+O+OH']-CovDat['Pt111+O']-E_g[0]+G_H2Ol
EOOHexpl7_5O = CovDat['Pt111+O+OOH']-CovDat['Pt111+O']-E_g[1]+G_H2Ol
EOHwO = np.mean(EOHexpl7_5O-EOHexpl7_5)
EOOHwO = np.mean(EOOHexpl7_5O-EOOHexpl7_5)

#obtaining coverage effects
#GCN values considered
GCNvals = [5.167,7.5,8.5]
Coverage = CovDat['Coverage']
EOHexpl8_5 = CovDat['Cavity+OH']-CovDat['Cavity']-E_g[0]+G_H2Ol
EOOHexpl8_5 = CovDat['Cavity+OOH']-CovDat['Cavity']-E_g[1]+G_H2Ol
EOHexpl5_167 = CovDat['Edge+OH']-CovDat['Edge']-E_g[0]+G_H2Ol
EOOHexpl5_167 = CovDat['Edge+OOH']-CovDat['Edge']-E_g[1]+G_H2Ol
OH7_5slope = np.linalg.lstsq(Coverage.as_matrix()[:,np.newaxis],(EOHexpl7_5 - EOHexpl7_5[0]))[0][0]
OOH7_5slope = np.linalg.lstsq(Coverage.as_matrix()[:,np.newaxis],(EOOHexpl7_5 - EOOHexpl7_5[0]))[0][0]
OH7_5Oslope = np.linalg.lstsq(Coverage.as_matrix()[:,np.newaxis],(EOHexpl7_5O - EOHexpl7_5O[0]))[0][0]
OOH7_5Oslope = np.linalg.lstsq(Coverage.as_matrix()[:,np.newaxis],(EOOHexpl7_5O - EOOHexpl7_5O[0]))[0][0]

#Buildig correlations of explicit solvation effect with BE
x = metal('Pt')
E7_5GCN = x.get_BEs(7.5)
E8_5GCN = x.get_BEs(8.5)
E5_167GCN = x.get_BEs(5.167)
EOHGCNimpl = np.array([E5_167GCN[0],E7_5GCN[0],E8_5GCN[0]]) + E_solv[0]
EOOHGCNimpl = np.array([E5_167GCN[1],E7_5GCN[1],E8_5GCN[1]]) + E_solv[1]
EOHexpl = np.array([EOHexpl5_167[0],EOHexpl7_5[0],EOHexpl8_5[0]])
EOOHexpl = np.array([EOOHexpl5_167[0],EOOHexpl7_5[0],EOOHexpl8_5[0]])
EOHimpl2explslope, EOHimpl2explint = np.polyfit(EOHGCNimpl, EOHexpl-EOHGCNimpl, 1)
EOOHimpl2explslope, EOOHimpl2explint = np.polyfit(EOOHGCNimpl, EOOHexpl-EOOHGCNimpl, 1)
print('OH impl2 expl slope and intercept')
print(EOHimpl2explslope,EOHimpl2explint)
print('OOH impl2 expl slope and intercept')
print(EOOHimpl2explslope, EOOHimpl2explint)

#Building correlations of coverage effects with BE
OH8_5slope = np.linalg.lstsq(Coverage.as_matrix()[:,np.newaxis],(EOHexpl8_5 - EOHexpl8_5[0]))[0][0]
OOH8_5slope = np.linalg.lstsq(Coverage.as_matrix()[:,np.newaxis],(EOOHexpl8_5 - EOOHexpl8_5[0]))[0][0]
OH5_167slope = np.linalg.lstsq(Coverage.as_matrix()[:,np.newaxis],(EOHexpl5_167 - EOHexpl5_167[0]))[0][0]
OOH5_167slope = np.linalg.lstsq(Coverage[0:5].as_matrix()[:,np.newaxis],(EOOHexpl5_167[0:5] - EOOHexpl5_167[0]))[0][0]
OHslopes = [OH5_167slope,OH7_5slope,OH8_5slope]
OOHslopes = [OOH5_167slope,OOH7_5slope,OOH8_5slope]
OHlowGCNslope, OHlowGCNint = np.polyfit(EOHGCNimpl[0:2],OHslopes[0:2],1)
OOHlowGCNslope, OOHlowGCNint = np.polyfit(EOOHGCNimpl[0:2],OOHslopes[0:2],1)
OHhighGCNslope, OHhighGCNint = np.polyfit(EOHGCNimpl[1:3],OHslopes[1:3],1)
OOHhighGCNslope, OOHhighGCNint = np.polyfit(EOOHGCNimpl[1:3],OOHslopes[1:3],1)
print('OH low GCN slope and intercept')
print(OHlowGCNslope, OHlowGCNint)
print('OOH low GCN slope and intercept')
print(OOHlowGCNslope, OOHlowGCNint)
print('OH high GCN slope and intercept')
print(OHhighGCNslope, OHhighGCNint)
print('OOH high GCN slope and intercept')
print(OOHhighGCNslope, OOHhighGCNint)