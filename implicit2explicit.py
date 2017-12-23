# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 22:36:33 2017

@author: lansf
"""
import os
from pandas import read_csv
data_fldr = '~/Box Sync/Synced_Files/Coding/Research/ORR Marcel/ORR_Honeycomb_Energies.csv'
data_fldr = os.path.expanduser(data_fldr)
E_H2Og = -14.219432

#Experimental value of H2O solvation from gas to liquid from:
#G. Schüürmann, M. Cossi, V. Barone, and J. Tomasi, The Journal of Physical Chemistry A 102, 6706 (1998).
#Experimental H2O solvation value confirmed by
#M. D. Liptak and G. C. Shields, J. Am. Chem. Soc. 123, 7314 (2001). and
#M. W. Palascak and G. C. Shields, The Journal of Physical Chemistry A 108, 3692 (2004).
Esolv_H2O = -0.2736
E_solv = [-0.575, -0.480] #OH* and OOH*
E_g = [-7.53, -13.26] #OH and OOH
Pt7_5 = [-208.21404,-218.12374,-222.52624] #P111 OH and OOH Energies without water
CovDat = read_csv(data_fldr)
E7H2O = -379.78779 # water in cavity
E6H2O = -365.04325 # removing H2O from cavity
Esolv_H2O_explicit = E7H2O-E6H2O-E_H2Og
G_H2Ol = E_H2Og + Esolv_H2O_explicit

#conversion factor from implicit to explicit adsorption energy
EOHimpl7_5 = Pt7_5[1] - Pt7_5[0] - E_g[0] + E_solv[0]
EOOHimpl7_5 = Pt7_5[2] - Pt7_5[0] - E_g[1] + E_solv[1]
EOHexpl7_5 = CovDat['Pt111+OH']-CovDat['Pt111']-E_g[0]+G_H2Ol
EOOHexpl7_5 = CovDat['Pt111+OOH']-CovDat['Pt111']-E_g[1]+G_H2Ol
#EOHexpl7_5 = CovDat['Pt111+OH']-CovDat['Pt111']-E_g[0]+G_H2Ol
#EOOHexpl7_5 = CovDat['Pt111+OOH']-CovDat['Pt111']-E_g[1]+G_H2Ol
EOHexpl7_5 = -10.89490338-E_g[0]
EOOHexpl7_5 = -14.61297244-E_g[1]
EOHimpl2expl = EOHexpl7_5-EOHimpl7_5
EOOHimpl2expl = EOOHexpl7_5-EOOHimpl7_5