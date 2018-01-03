# -*- coding: utf-8 -*-
"""
Determines the explicit solvation correction when compared to the implicit value.
The explicit value is taken at zero coverage from G_fit.py

@author: lansf
"""

#Experimental value of H2O solvation from gas to liquid from:
#G. Schüürmann, M. Cossi, V. Barone, and J. Tomasi, The Journal of Physical Chemistry A 102, 6706 (1998).
#Experimental H2O solvation value confirmed by
#M. D. Liptak and G. C. Shields, J. Am. Chem. Soc. 123, 7314 (2001). and
#M. W. Palascak and G. C. Shields, The Journal of Physical Chemistry A 108, 3692 (2004).
E_solv = [-0.575, -0.480] #OH* and OOH*
E_g = [-7.53, -13.26] #OH and OOH
Pt7_5 = [-208.21404,-218.12374,-222.52624] #P111 OH and OOH Energies without water

#conversion factor from implicit to explicit adsorption energy
EOHimpl7_5 = Pt7_5[1] - Pt7_5[0] - E_g[0] + E_solv[0]
EOOHimpl7_5 = Pt7_5[2] - Pt7_5[0] - E_g[1] + E_solv[1]
EOHexpl7_5 = -10.89490338-E_g[0]
EOOHexpl7_5 = -14.61297244-E_g[1]
EOHimpl2expl = EOHexpl7_5-EOHimpl7_5
EOOHimpl2expl = EOOHexpl7_5-EOOHimpl7_5

"""print impl 2 expl conversion"""
print(EOHimpl2expl)
print(EOOHimpl2expl)
