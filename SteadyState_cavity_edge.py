# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 11:05:24 2017

@author: lansf
"""
import numpy as np

def ORR_rate(Theta,t,dGdOHedge,dGdOHcav,dGdOOHedge,dGdOOHcav,dGdO,sO,tpO,uO,xO,yO,zO,aO,bO,cO,x,x2,x3,y,z,a,b,c,d,U):
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
    
    OHedge = Theta[0]; OHcav = Theta[1]; OOHedge = Theta[2]; OOHcav = Theta[3]
    Ocovfcc = Theta[4]; Ocovatop = Theta[5]

    #Calculating Coverage Dependent Adsorption Energies   
    dE_OHedge = dGdOHedge(np.array([OHedge,OHcav,OOHedge,OOHcav,(Ocovfcc+Ocovatop)]),x,x2,x3,y,z,a)
    dE_OHcav = dGdOHcav(np.array([OHedge,OHcav,OOHedge,OOHcav,(Ocovfcc+Ocovatop)]),b,x2)
    dE_OOHedge = dGdOOHedge(np.array([OHedge,OHcav,OOHedge,OOHcav,(Ocovfcc+Ocovatop)]),x,x2,x3,y,z,c)
    dE_OOHcav = dGdOOHcav(np.array([OHedge,OHcav,OOHedge,OOHcav,(Ocovfcc+Ocovatop)]),d,x3)
    dE_Ofcc = dGdO(np.array([(OHedge+OHcav),(Ocovfcc+Ocovatop),(OOHedge+OOHcav)]),sO,tpO,uO,xO,yO,zO,bO)
    
    
    # Species free energies at T = 298K
    G_OHedge = dE_OHedge + ZPE[0] - TS[0] #G minus G of surface
    G_OOHedge = dE_OOHedge + ZPE[1] - TS[1] # G minus G of surface
    G_OHcav = dE_OHcav + ZPE[0] - TS[0] #G minus G of surface
    G_OOHcav = dE_OOHcav + ZPE[1] - TS[1] # G minus G of surface
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
    
    G1edge = G_OOHedge - G_O2g - G_H_e
    G1cav = G_OOHcav - G_O2g - G_H_e
    G2edge = G_Ofcc + G_H2Ol - G_OOHedge - G_H_e
    G2cav = G_Ofcc + G_H2Ol - G_OOHcav - G_H_e
    G2aedge = G_Oatop + G_H2Ol - G_OOHedge - G_H_e
    G2acav = G_Oatop + G_H2Ol - G_OOHcav - G_H_e
    G2bedge = G_Ofcc + G_OHedge - G_OOHedge
    G2bcav = G_Ofcc + G_OHcav - G_OOHcav
    G3edge = G_OHedge - G_Ofcc - G_H_e
    G3cav = G_OHcav - G_Ofcc - G_H_e
    G3aedge = G_OHedge - G_Oatop - G_H_e
    G3acav = G_OHcav - G_Oatop - G_H_e
    G4edge = G_H2Ol - G_OHedge - G_H_e
    G4cav = G_H2Ol - G_OHcav - G_H_e
    G_O2fcc = 2*G_Ofcc - G_O2g
    #print('G1:',G1,'G2:',G2,'G2a:',G2a,'G2b:',G2b,'G3:',G3,'G3a:',G3a,'G4:',G4)
      
    Ea1 = 0.07 # O2 protonation barrier from Hyman 2006
    k1edge = kB*T/h*np.exp(-max(G1edge+Ea1,Ea1)/(kB*T))
    k_1edge = kB*T/h*np.exp(-max(-G1edge+Ea1,Ea1)/(kB*T))
    k1cav = kB*T/h*np.exp(-max(G1cav+Ea1,Ea1)/(kB*T))
    k_1cav = kB*T/h*np.exp(-max(-G1cav+Ea1,Ea1)/(kB*T))
    Ea2 = 0.01 + 0.14 #OH protonation and diffusion from Hyman 2006 in place of OOH protonation and O diffusion
    k2edge = kB*T/h*np.exp(-max(G2edge+Ea2,Ea2)/(kB*T))
    k_2edge = kB*T/h*np.exp(-max(-G2edge+Ea2,Ea2)/(kB*T))
    k2cav = kB*T/h*np.exp(-max(G2cav+Ea2,Ea2)/(kB*T))
    k_2cav = kB*T/h*np.exp(-max(-G2cav+Ea2,Ea2)/(kB*T))
    Ea2a = 0.01 #OH protonation in place of OOH protonation
    k2aedge = kB*T/h*np.exp(-max(G2aedge+Ea2a,Ea2a)/(kB*T))
    k_2aedge = kB*T/h*np.exp(-max(-G2aedge+Ea2a,Ea2a)/(kB*T))
    k2acav = kB*T/h*np.exp(-max(G2acav+Ea2a,Ea2a)/(kB*T))
    k_2acav = kB*T/h*np.exp(-max(-G2acav+Ea2a,Ea2a)/(kB*T))
    Ea2b = 0.22 #OOH dissociation from Hyman 2006
    k2bedge = kB*T/h*np.exp(-max(G2bedge+Ea2b,Ea2b)/(kB*T))
    k_2bedge = kB*T/h*np.exp(-max(-G2bedge+Ea2b,Ea2b)/(kB*T))
    k2bcav = kB*T/h*np.exp(-max(G2bcav+Ea2b,Ea2b)/(kB*T))
    k_2bcav = kB*T/h*np.exp(-max(-G2bcav+Ea2b,Ea2b)/(kB*T))
    Ea3 = 0.03 + 0.14 #O protonation and OH diffusion in from Hyman 2006
    k3edge = kB*T/h*np.exp(-max(G3edge+Ea3,Ea3)/(kB*T))
    k_3edge = kB*T/h*np.exp(-max(-G3edge+Ea3,Ea3)/(kB*T))
    k3cav = kB*T/h*np.exp(-max(G3cav+Ea3,Ea3)/(kB*T))
    k_3cav = kB*T/h*np.exp(-max(-G3cav+Ea3,Ea3)/(kB*T))
    Ea3a = 0.03 #O protonation from Hyman 2006
    k3aedge = kB*T/h*np.exp(-max(G3aedge+Ea3a,Ea3a)/(kB*T))
    k_3aedge = kB*T/h*np.exp(-max(-G3aedge+Ea3a,Ea3a)/(kB*T))
    k3acav = kB*T/h*np.exp(-max(G3acav+Ea3a,Ea3a)/(kB*T))
    k_3acav = kB*T/h*np.exp(-max(-G3acav+Ea3a,Ea3a)/(kB*T))
    Ea4 = 0.01 # OH protonation from Hyman 2006
    k4edge = kB*T/h*np.exp(-max(G4edge+Ea4,Ea4)/(kB*T))
    k_4edge = kB*T/h*np.exp(-max(-G4edge+Ea4,Ea4)/(kB*T))
    k4cav = kB*T/h*np.exp(-max(G4cav+Ea4,Ea4)/(kB*T))
    k_4cav = kB*T/h*np.exp(-max(-G4cav+Ea4,Ea4)/(kB*T))
    EaO2 = 0.65 #dissociation barrier for O2 from Yan 2017
    kO2fcc = kB*T/h*np.exp(-max(G_O2fcc+EaO2,EaO2)/(kB*T))
    k_O2fcc = kB*T/h*np.exp(-max(-G_O2fcc+EaO2,EaO2)/(kB*T))
    
    
    r1edge=k1edge*(1-OHedge-OOHedge-Ocovatop)*pO2*pH2**0.5
    r_1edge = k_1edge*OOHedge
    r1cav=k1cav*(1-OHcav-OOHcav-Ocovatop)*pO2*pH2**0.5
    r_1cav = k_1cav*OOHcav
    r2edge = k2edge*OOHedge*pH2**0.5
    r_2edge = k_2edge*Ocovfcc*pH2O
    r2cav = k2cav*OOHcav*pH2**0.5
    r_2cav = k_2cav*Ocovfcc*pH2O
    r2aedge = k2aedge*OOHedge*pH2**0.5
    r_2aedge = k_2aedge*Ocovatop*pH2O
    r2acav = k2acav*OOHcav*pH2**0.5
    r_2acav = k_2acav*Ocovatop*pH2O
    r2bedge = k2bedge*OOHedge
    r_2bedge = k_2bedge*Ocovfcc*OHedge
    r2bcav = k2bcav*OOHcav
    r_2bcav = k_2bcav*Ocovfcc*OHcav
    r3edge = k3edge*Ocovfcc*pH2**0.5
    r_3edge = k_3edge*OHedge
    r3cav = k3cav*Ocovfcc*pH2**0.5
    r_3cav = k_3cav*OHcav
    r3aedge = k3aedge*Ocovatop*pH2**0.5
    r_3aedge = k_3aedge*OHedge
    r3acav = k3acav*Ocovatop*pH2**0.5
    r_3acav = k_3acav*OHcav
    r4edge = k4edge*OHedge*pH2**0.5
    r_4edge = k_4edge*(1-OHedge-OOHedge-Ocovatop)*pH2O
    r4cav = k4cav*OHcav*pH2**0.5
    r_4cav = k_4cav*(1-OHcav-OOHcav-Ocovatop)*pH2O
    rOfcc = 2*(kO2fcc*pO2*2*(1-Ocovfcc)**2)
    r_Ofcc = 2*(k_O2fcc*2*(Ocovfcc)**2)
    
    dThetaOOHedgedt = r1edge - r_1edge - r2edge + r_2edge - r2aedge + r_2aedge - r2bedge + r_2bedge
    dThetaOHedgedt = r2bedge - r_2bedge + r3edge - r_3edge + r3aedge - r_3aedge - r4edge + r_4edge #+ r3b - r_3b
    dThetaOOHcavdt = r1cav - r_1cav - r2cav + r_2cav - r2acav + r_2acav - r2bcav + r_2bcav
    dThetaOHcavdt = r2bcav - r_2bcav + r3cav - r_3cav + r3acav - r_3acav - r4cav + r_4cav #+ r3b - r_3b
    r2 = r2edge+r2cav; r_2 = r_2edge+r_2cav; r2a = r2aedge+r2acav; r_2a = r_2aedge+r_2acav
    r2b = r2bedge+r2bcav; r_2b = r_2bedge+r_2bcav; r3=r3edge+r3cav; r_3=r_3edge+r_3cav; 
    r3a = r3aedge+r3acav; r_3a=r_3aedge+r_3acav
    
    #r2 = r2cav; r_2 = r_2cav; r2a = r2acav; r_2a = r_2acav
    #r2b = r2bcav; r_2b = r_2bcav; r3=r3cav; r_3=r_3cav; 
    #r3a = r3acav; r_3a=r_3acav
    
    #r2 = r2edge; r_2 = r_2edge; r2a = r2aedge; r_2a = r_2aedge
    #r2b = r2bedge; r_2b = r_2bedge; r3=r3edge; r_3=r_3edge;
    #r3a = r3aedge; r_3a=r_3aedge
    
    dThetaOfccdt = rOfcc - r_Ofcc + r2 - r_2 + r2b - r_2b - r3 + r_3 
    dThetaOatopdt = r2a - r_2a - r3a + r_3a # - r3b + r_3b
    
    dydt = [dThetaOHedgedt,dThetaOHcavdt,dThetaOOHedgedt,dThetaOOHcavdt,dThetaOfccdt,dThetaOatopdt]
    return dydt
