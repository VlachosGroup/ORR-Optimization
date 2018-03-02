# -*- coding: utf-8 -*-
"""
Determines the explicit solvation correction when compared to the implicit value.
The explicit value is taken at zero coverage from G_fit.py
@author: lansf
"""

Pt_impl = [-208.21404,-218.12374,-222.52624,-214.35223] #P111 OH and OOH, and O Energies without water for a 3x3 slab
Pt_expl = [-385.51603,-381.21702,-385.40342,-391.64417] #P111 OH and OOH, and O Energies with 2 layers of water for a 3x3 slab
E7H2O = -379.78779 # water in cavity
E6H2O = -365.04325 # removing H2O from cavity
EH2Oexpl = E7H2O-E6H2O #this is the energy of H2O interacting with a surface
#conversion factor from implicit to explicit adsorption energy
EOHimpl = Pt_impl[1] - Pt_impl[0]
EOOHimpl = Pt_impl[2] - Pt_impl[0]
EOimpl = Pt_impl[3] - Pt_impl[0]
EOHexpl = Pt_expl[1] - Pt_expl[0] + EH2Oexpl
EOOHexpl = Pt_expl[2] - Pt_expl[0] + EH2Oexpl
EOexpl = Pt_expl[3] - Pt_expl[0]

dEOHexpl = EOHexpl - EOHimpl
dEOOHexpl = EOOHexpl - EOOHimpl
dEOexpl = EOexpl - EOimpl

"""explicit solvation energy"""
print(dEOHexpl)
print(dEOOHexpl)
print(dEOexpl)