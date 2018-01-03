# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 11:05:24 2017

@author: lansf
"""
import numpy as np

def ORR_rate(Theta,t,dGdOH, dGdOOH,dGdO,s,tp,u,x,y,z,a,b,c,U):
    kB = 8.617e-5                      # eV / K
    h = 4.135667662e-15;               # eV * s
    T = 298.15                         # K
    U_0 = 1.23                         # eV, theoretical maximum cell voltage for ORR
    U = U                            # V, cathode potential
    pO2g = 1; pH2 = 1; pH2O = 1         #Pressures of O2, H2 and H2O [atm]
    hO2 = 0.0013                        #Henry's constant in mol/(kg*bar)
    kg2mol = 55.5                   #moles of water in 1 kg H2O
    pO2 = hO2*pO2g/kg2mol
    # This is empirically fitted to match the current density of Pt(111) from experiments
    n = 1                               # number of electrons tranfered in each step    
    
    # *OH, *OOH, O*
    ZPE = [0.332, 0.428, 0.072]                # zero-point energy correction, eV
    TS = [0, 0, 0]                         # entropy contribution to Gibbs energy at 298 K, eV
    
    #Getting Coverages
    
    OHcov = Theta[0]; OOHcov = Theta[1]; Ocovfcc = Theta[2]; Ocovatop = Theta[3]

    #Calculating Coverage Dependent Adsorption Energies   
    dE_OH = dGdOH(np.array([OHcov,OOHcov,Ocovfcc]),s,tp,u,x,y,z,a)
    dE_Ofcc = dGdO(np.array([OHcov,OOHcov,Ocovfcc]),s,tp,u,x,y,z,b)
    dE_OOH = dGdOOH(np.array([OHcov,OOHcov,Ocovfcc]),x,y,z,c)
    
    # Species free energies at T = 298K
    G_OH = dE_OH + ZPE[0] - TS[0] #G minus G of surface
    G_OOH = dE_OOH + ZPE[1] - TS[1] # G minus G of surface
    #RPBE-PBE G_O = 0.316
    G_Ofcc = dE_Ofcc + ZPE[2] - TS[2]
    G_Oatop = G_Ofcc + -212.88971 - -214.35223
    
    
    # Gas species Gibbs energies
    # H2(g), H2O(l), O2(g), OH(g), OOH(g), O2 (g)
    E_DFT_gas = [-6.7595, -14.2222, -9.86]             # From my own DFT data
    
    # H2, H2O(l), O2(gas)
    ZPE_gas = [0.270, 0.574, 0.0971]             # eV, 
    TS_gas = [0.404, 0.583, 0.634]              # at 298 K, eV / K
    E_solv_gas = [0, -0.087]             # eV
    
    G_H2g = E_DFT_gas[0] + ZPE_gas[0] - TS_gas[0] + E_solv_gas[0]
    G_H2Ol = E_DFT_gas[1] + ZPE_gas[1] - TS_gas[1] + E_solv_gas[1]
    G_O2g = 2 * (G_H2Ol - G_H2g) + 4 * U_0
    G_H_e = 0.5*G_H2g - U*n
    
    G1 = G_OOH - G_O2g - G_H_e
    G2 = G_Ofcc + G_H2Ol - G_OOH - G_H_e
    G2a = G_Oatop + G_H2Ol - G_OOH - G_H_e
    G2b = G_Ofcc + G_OH - G_OOH
    G3 = G_OH - G_Ofcc - G_H_e
    G3a = G_OH - G_Oatop - G_H_e
    G4 = G_H2Ol - G_OH - G_H_e
    G_O2fcc = 2*G_Ofcc - G_O2g
    #print('G1:',G1,'G2:',G2,'G2a:',G2a,'G2b:',G2b,'G3:',G3,'G3a:',G3a,'G4:',G4)
      
    Ea1 = 0.07 # O2 protonation barrier from Hyman 2006
    k1 = kB*T/h*np.exp(-max(G1+Ea1,Ea1)/(kB*T))
    k_1 = kB*T/h*np.exp(-max(-G1+Ea1,Ea1)/(kB*T))
    Ea2 = 0.01 + 0.14 #OH protonation and diffusion from Hyman 2006 in place of OOH protonation and O diffusion
    k2 = kB*T/h*np.exp(-max(G2+Ea2,Ea2)/(kB*T))
    k_2 = kB*T/h*np.exp(-max(-G2+Ea2,Ea2)/(kB*T))
    Ea2a = 0.01 #OH protonation in place of OOH protonation
    k2a = kB*T/h*np.exp(-max(G2a+Ea2a,Ea2a)/(kB*T))
    k_2a = kB*T/h*np.exp(-max(-G2a+Ea2a,Ea2a)/(kB*T))
    Ea2b = 0.22 #OOH dissociation from Hyman 2006
    k2b = kB*T/h*np.exp(-max(G2b+Ea2b,Ea2b)/(kB*T))
    k_2b = kB*T/h*np.exp(-max(-G2b+Ea2b,Ea2b)/(kB*T))
    Ea3 = 0.03 + 0.14 #O protonation and OH diffusion in from Hyman 2006
    k3 = kB*T/h*np.exp(-max(G3+Ea3,Ea3)/(kB*T))
    k_3 = kB*T/h*np.exp(-max(-G3+Ea3,Ea3)/(kB*T))
    Ea3a = 0.03 #O protonation from Hyman 2006
    k3a = kB*T/h*np.exp(-max(G3a+Ea3a,Ea3a)/(kB*T))
    k_3a = kB*T/h*np.exp(-max(-G3a+Ea3a,Ea3a)/(kB*T))
    Ea4 = 0.01 # OH protonation from Hyman 2006
    k4 = kB*T/h*np.exp(-max(G4+Ea4,Ea4)/(kB*T))
    k_4 = kB*T/h*np.exp(-max(-G4+Ea4,Ea4)/(kB*T))
    EaO2 = 0.65 #dissociation barrier for O2 from Yan 2017
    kO2fcc = kB*T/h*np.exp(-max(G_O2fcc+EaO2,EaO2)/(kB*T))
    k_O2fcc = kB*T/h*np.exp(-max(-G_O2fcc+EaO2,EaO2)/(kB*T))
    
    
    r1=k1*(1-OHcov-OOHcov-Ocovatop)*pO2*pH2**0.5
    r_1 = k_1*OOHcov
    r2 = k2*OOHcov*pH2**0.5
    r_2 = k_2*Ocovfcc*pH2O
    r2a = k2a*OOHcov*pH2**0.5
    r_2a = k_2a*Ocovatop*pH2O
    r2b = k2b*OOHcov
    r_2b = k_2b*Ocovfcc*OHcov
    r3 = k3*Ocovfcc*pH2**0.5
    r_3 = k_3*OHcov
    r3a = k3a*Ocovatop*pH2**0.5
    r_3a = k_3a*OHcov
    r4 = k4*OHcov*pH2**0.5
    r_4 = k_4*(1-OHcov-OOHcov-Ocovatop)*pH2O
    rOfcc = 2*(kO2fcc*pO2*2*(1-Ocovfcc)**2)
    r_Ofcc = 2*(k_O2fcc*2*(Ocovfcc)**2)
    
    dThetaOOHdt = r1 - r_1 - r2 + r_2 - r2a + r_2a - r2b + r_2b
    dThetaOHdt = r2b - r_2b + r3 - r_3 + r3a - r_3a - r4 + r_4
    dThetaOfccdt = rOfcc - r_Ofcc + r2 - r_2 + r2b - r_2b - r3 + r_3 
    dThetaOatopdt = r2a - r_2a - r3a + r_3a
    dydt = [dThetaOHdt,dThetaOOHdt,dThetaOfccdt,dThetaOatopdt]
    return dydt
