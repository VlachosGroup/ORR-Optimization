'''
Computes the rate of the oxygen reduction reaction
'''


import numpy as np

def ORR_rate(delEads_OH, delEads_OOH,explicit=False):
    
    '''
    Compute ORR rate from OH and OOH binding energies
    Follows the method and data in S.1 of F. Calle-Vallejo et al., Science 350(6257), 185 (2015).
    Also see J. K. Norskov et al., The Journal of Physical Chemistry B 108(46), 17886 (2004).
    for method on how to convert binding energies to activity
    
    :param delEads_OH: low coverage DFT binding energy of OH
    :param delEads_OOH: low coverage DFT binding energy of OOH
    :optional param explicit: converts from implicit to explicit solvation, default=False
    :optional param oxygen: scales energies for presence of oxygen (1/9 ML atomic O), default=False
    :optional param coverage (ML): scales energies for lateral interactions default=0
    :optional param variable: incorporates GCN dependendence of the explicit
    solvation and coverage effects through the dependence on implicit energy (default=False)
    :returns: Current [miliAmperes (mA) per atom]
    '''        
    kB = 8.617e-5                      # eV / K
    T = 298.15                         # K
    U_0 = 1.23                         # eV, theoretical maximum cell voltage for ORR
    U = 0.9                             # V, cathode potential
    i_c = 1.6123434625e-12                    # miliAmperes (mA) per atom, divide by surface area to get current density
        # This is empirically fitted to match the current density of Pt(111) from experiments
    n = 1                               # number of electrons tranfered in each step    
    
    # *OH, *OOH
    E_g = [-7.53, -13.26]               # energies of OH(g) and OOH(g)
    ZPE = [0.332, 0.428]                # zero-point energy correction, eV
    TS = [0, 0]                         # entropy contribution to Gibbs energy at 298 K, eV
    E_solv = [-0.575, -0.480]           # solvation energy, eV

    #add implicit solvation energy
    delEads_OH += E_solv[0]
    delEads_OOH += E_solv[1]
    
    
    
    #exchange implicit for explicit solvation effects
    if explicit == True:
        EOHimpl2expl = 0.03917; EOOHimpl2expl = 0.16027
        delEads_OH += EOHimpl2expl
        delEads_OOH += EOOHimpl2expl
    
    # Species free energies at T = 298K
    G_OH = E_g[0] + delEads_OH + ZPE[0] - TS[0]
    G_OOH = E_g[1] + delEads_OOH + ZPE[1] - TS[1]
    
    # Gas species Gibbs energies
    # H2(g), H2O(l), O2(g), OH(g), OOH(g)
    E_DFT_gas = [-6.7595, -14.2222, -9.86]             # From my own DFT data
    
    
    # H2, H2O(l)
    ZPE_gas = [0.270, 0.574]             # eV, 
    TS_gas = [0.404, 0.583]              # at 298 K, eV / K
    E_solv_gas = [0, -0.087]             # eV
    
    G_H2g = E_DFT_gas[0] + ZPE_gas[0] - TS_gas[0] + E_solv_gas[0]
    G_H2Ol = E_DFT_gas[1] + ZPE_gas[1] - TS_gas[1] + E_solv_gas[1]
    G_O2g = 2 * (G_H2Ol - G_H2g) + 4 * U_0
    
    # These are the values you should get for gas phase Gibbs energies
    # G_H2g = -6.8935
    # G_H2Ol = -14.3182
    # G_O2g = -9.9294
    
    # Compute G1 and G4 - without referencing
    #G1 = G_OOH - G_O2g - 0.5 * G_H2g
    #G4 = G_H2Ol - G_OH - 0.5 * G_H2g
    
    # Compute G1 and G4 - with referencing to H2(g) and H2O(l)
    delta_G_OH = G_OH + 0.5 * G_H2g - G_H2Ol
    delta_G_OOH = G_OOH + 1.5 * G_H2g - 2*G_H2Ol
    delta_G_O2 = 4 * U_0
    G1 = delta_G_OOH - delta_G_O2
    G4 = - delta_G_OH
    
    # Add contribution from cell potential
    G1 += U * n
    G4 += U * n
    
    delG_max = max(G1,G4)
    j = i_c * np.exp( - delG_max  / (kB * T) )
    
    # Check which step is rate determining
    is_step_4 = G1 < G4;          # *OH desorption (Step 4) is rate determining
    if is_step_4:
        RDS = 4
    else:
        RDS = 1
    
    return j