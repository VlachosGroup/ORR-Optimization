# -*- coding: utf-8 -*-
"""
Created on Sat May  4 16:15:14 2019

@author: mpnun
"""

'''
Extract the data
'''

import os
import numpy as np
from orr_optimizer.orr_cat import *

metal_list = ['Pt', 'Au']
facet_list = ['111', '100']

'''
User section
'''

# Point this to where you have the data (from Josh's email)
data_fldr = 'C:\\Users\\mpnun\\Desktop\\FromRepository\\7_6_2016'

# Use any of the 4 combinations of metal and facet
met = metal_list[0]
fac = facet_list[0]

'''
End user section
'''

cat = orr_cat(dim1 = 30, dim2 = 30, volcano = 'CV')
cat.randomize(build_structure = True)


data_fldr2 = os.path.join(data_fldr, met, fac)


for i in range(250):
    fldr3 = os.path.join(data_fldr2,str(i+1),'quench','optimum.bin')
    a = np.fromfile(fldr3, dtype=np.uint32)
    
    variable_occs = a[-900::]                                   # This is 'x'
    cat.assign_occs(variable_occs)
    current = cat.eval_current_density(normalize = True)        # current density in mA/cm^2
    surf_eng = cat.eval_surface_energy(normalize = True)        # surface energy in J/m^2
    print([current, surf_eng])
    