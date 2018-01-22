# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 11:05:24 2017

@author: lansf
"""
from __future__ import division
import os
from pandas import read_csv
import numpy as np
from scipy.optimize import curve_fit
from scipy.integrate import odeint
from metal import metal


class ORR_MKM:
    def __init__(self, site_type):
        self.site_type = site_type
        E_H2Og = -14.219432
        E7H2O = -379.78779 # water in cavity
        E6H2O = -365.04325 # removing H2O from cavity
        Esolv_H2O_explicit = E7H2O-E6H2O-E_H2Og #this is solvation energy of H2O interacting with a surface
        self.G_H2Osurf = E_H2Og + Esolv_H2O_explicit
        self.Gfit_terrace()
        self.Gfit_cavity_edge()

    def Gfit_terrace(self):
        data_file = 'Surface_Energies.csv'
        data_file = os.path.expanduser(data_file)
        CovDat = read_csv(data_file)
        Coverages = np.array([CovDat.OHcov,CovDat.OOHcov,CovDat.Ocov])
        WaterReplacement = np.sum(CovDat[['OHcov','OOHcov']],axis=1)*9*self.G_H2Osurf
        def Gsurf(Coverageinput,s,tp,u,x,y,z,GOHo,GOOHo,GOo):
            OHcov, OOHcov, Ocov = Coverageinput
            Gval = GOHo*OHcov + GOOHo*OOHcov + GOo*Ocov + s*(tp*Ocov+OHcov)**u + x*(y*Ocov+OHcov)**z*OOHcov
            return Gval
        Go = -385.40342
        Energies = CovDat.Energy.as_matrix() + WaterReplacement - Go
        lmin = 0
        lmax = 30
        emin = 1
        emax=4
        self.popt, pcov = curve_fit(Gsurf,Coverages,Energies/9.0,bounds=(np.array([lmin,lmin,emin,lmin,lmin,emin,-20,-20,-20]),np.array([lmax,lmax,emax,lmax,lmax,emax,0,0,0])))
        def dGdOH(Coverageinput,popt,GCN_scaling):
            s,tp,u,x,y,z,GOHo,GOOHo,GOo = popt
            Coverageinput = [i if i>0 else 0 for i in Coverageinput]
            OHcov, OOHcov, Ocov = Coverageinput
            dGval = GOHo*GCN_scaling + u*s*(tp*Ocov+OHcov)**(u-1) + z*x*(y*Ocov+OHcov)**(z-1)*OOHcov
            return dGval
                    
        def dGdOOH(Coverageinput,popt,GCN_scaling):
            s,tp,u,x,y,z,GOHo,GOOHo,GOo = popt
            Coverageinput = [i if i>0 else 0 for i in Coverageinput]
            OHcov, OOHcov, Ocov = Coverageinput
            dGval = GOOHo*GCN_scaling + x*(y*Ocov+OHcov)**z
            return dGval
        
        def dGdO(Coverageinput,popt):
            s,tp,u,x,y,z,GOHo,GOOHo,GOo = popt
            Coverageinput = [i if i>0 else 0 for i in Coverageinput]
            OHcov, OOHcov, Ocov = Coverageinput
            dGval = GOo + tp*u*s*(tp*Ocov+OHcov)**(u-1)+y*z*x*(y*Ocov+OHcov)**(z-1)*OOHcov
            return dGval

        self.dGdOH = dGdOH
        self.dGdOOH = dGdOOH
        self.dGdO = dGdO
        
    def Gfit_cavity_edge(self):
        data_file = 'Surface_Energies_cavity.csv'
        data_file = os.path.expanduser(data_file)
        CovDat = read_csv(data_file)
        Coverages = np.array([CovDat.OH_edge,CovDat.OH_cavity,CovDat.OOH_edge,CovDat.OOH_cavity])
        WaterReplacement = np.sum(Coverages,axis=0)*9*self.G_H2Osurf
        def Gsurf(Coverageinput,x,x2,x3,y,z,GOHedgeo,GOHcavo,GOOHedgeo,GOOHcavo):
            OHedge, OHcav, OOHedge, OOHcav = Coverageinput
            Gval = GOHedgeo*OHedge + GOHcavo*OHcav + GOOHedgeo*OOHedge + GOOHcavo*OOHcav + x*(y*OHedge+OOHedge)**z + x2*(OHedge+OOHedge)*OHcav + x3*(OHedge+OOHedge)*OOHcav
            return Gval
        
        Go = -365.04325
        Energies = CovDat.Energy.as_matrix() + WaterReplacement - Go
        lmin = 0
        lmax = 30
        emin = 1
        emax=4
        self.popt_cavity_edge, pcov = curve_fit(Gsurf,Coverages,Energies/9.0,bounds=(np.array([lmin,lmin,lmin,lmin,emin,-20,-20,-20,-20]),np.array([lmax,lmax,lmax,lmax,emax,0,0,0,0])))
        def dGdOHedge(Coverageinput,popt,popt_terrace,GCN_scaling):
            x,x2,x3,y,z,GOHedgeo,GOHcavo,GOOHedgeo,GOOHcavo = popt
            (s_terrace,tp_terrace,u_terrace,x_terrace,y_terrace,z_terrace
            ,GOHo_terrace,GOOHo_terrace,GOo_terrace) = popt_terrace
            Coverageinput = [i if i>0 else 0 for i in Coverageinput]
            OHedge, OHcav, OOHedge, OOHcav, Ocov = Coverageinput
            dGval = GOHedgeo*GCN_scaling + y*x*z*(y*OHedge+OOHedge)**(z-1) + x2*OHcav + x3*OOHcav
            + u_terrace*s_terrace*(tp_terrace*Ocov)**(u_terrace-1)
            return dGval

        def dGdOHcav(Coverageinput,popt,popt_terrace,GCN_scaling):
            x,x2,x3,y,z,GOHedgeo,GOHcavo,GOOHedgeo,GOOHcavo = popt
            (s_terrace,tp_terrace,u_terrace,x_terrace,y_terrace,z_terrace
            ,GOHo_terrace,GOOHo_terrace,GOo_terrace) = popt_terrace
            Coverageinput = [i if i>0 else 0 for i in Coverageinput]
            OHedge, OHcav, OOHedge, OOHcav, Ocov = Coverageinput
            dGval = GOHcavo*GCN_scaling + x2*(OHedge+OOHedge)
            + u_terrace*s_terrace*(tp_terrace*Ocov)**(u_terrace-1)
            return dGval
        
        def dGdOOHedge(Coverageinput,popt,popt_terrace,GCN_scaling):
            x,x2,x3,y,z,GOHedgeo,GOHcavo,GOOHedgeo,GOOHcavo = popt
            (s_terrace,tp_terrace,u_terrace,x_terrace,y_terrace,z_terrace
            ,GOHo_terrace,GOOHo_terrace,GOo_terrace) = popt_terrace
            Coverageinput = [i if i>0 else 0 for i in Coverageinput]
            OHedge, OHcav, OOHedge, OOHcav, Ocov = Coverageinput
            dGval = GOOHedgeo*GCN_scaling + x*z*(y*OHedge+OOHedge)**(z-1) + x2*OHcav + x3*OOHcav
            + x_terrace*(y_terrace*Ocov)**z_terrace
            return dGval
        
        def dGdOOHcav(Coverageinput,popt,popt_terrace,GCN_scaling):
            x,x2,x3,y,z,GOHedgeo,GOHcavo,GOOHedgeo,GOOHcavo = popt
            (s_terrace,tp_terrace,u_terrace,x_terrace,y_terrace,z_terrace
            ,GOHo_terrace,GOOHo_terrace,GOo_terrace) = popt_terrace
            Coverageinput = [i if i>0 else 0 for i in Coverageinput]
            OHedge, OHcav, OOHedge, OOHcav,Ocov = Coverageinput
            dGval = GOOHcavo*GCN_scaling + x3*(OHedge + OOHedge)
            + x_terrace*(y_terrace*Ocov)**z_terrace
            return dGval
        self.dGdOHedge = dGdOHedge
        self.dGdOHcav = dGdOHcav
        self.dGdOOHedge = dGdOOHedge
        self.dGdOOHcav = dGdOOHcav
    
    def coverage_terrace(self,Theta,t,popt,GCN_scaling):
        kB = 8.617e-5                      # eV / K
        h = 4.135667662e-15;               # eV * s
        T = 298.15                         # K
        U_0 = 1.23                         # eV, theoretical maximum cell voltage for ORR
        U = 0.9                            # V, cathode potential
        pO2g = 1; pH2 = 1; pH2O = 1         #Pressures of O2, H2 and H2O [atm]
        hO2 = 0.0013                        #Henry's constant in mol/(kg*bar)
        kg2mol = 55.5                   #moles of water in 1 kg H2O
        pO2 = hO2*pO2g/kg2mol
        n = 1                               # number of electrons tranfered in each step    
        # *OH, *OOH, O*
        ZPE = [0.332, 0.428, 0.072]                # zero-point energy correction, eV
        TS = [0, 0, 0]                         # entropy contribution to Gibbs energy at 298 K, eV
        #Getting Coverages
        OHcov = Theta[0]; OOHcov = Theta[1]; Ocovfcc = Theta[2]; Ocovatop = Theta[3]
        #Calculating Coverage Dependent Adsorption Energies   
        dE_OH = self.dGdOH(np.array([OHcov,OOHcov,Ocovfcc]),popt,GCN_scaling[0])
        dE_Ofcc = self.dGdO(np.array([OHcov,OOHcov,Ocovfcc]),popt)
        dE_OOH = self.dGdOOH(np.array([OHcov,OOHcov,Ocovfcc]),popt,GCN_scaling[1])
        # Surface Species free energies at T = 298K
        G_OH = dE_OH + ZPE[0] - TS[0] #G minus G of surface
        G_OOH = dE_OOH + ZPE[1] - TS[1] # G minus G of surface
        G_Ofcc = dE_Ofcc + ZPE[2] - TS[2] #RPBE-PBE G_O = 0.316
        G_Oatop = G_Ofcc + -212.88971 - -214.35223
        # Gas species Gibbs energies
        # H2(g), H2O(l), O2(g), OH(g), OOH(g), O2 (g)
        E_DFT_gas = [-6.7595, -14.2222, -9.86] # From my own DFT data
        # H2, H2O(l), O2(gas)
        ZPE_gas = [0.270, 0.574, 0.0971]  # eV 
        TS_gas = [0.404, 0.583, 0.634]  # at 298 K, eV / K
        E_solv_gas = [0, -0.087]  # eV H2O(l) solvation if TS(g) at 298K
        """Computing Gibbs energies of gas and solvated species"""
        G_H2g = E_DFT_gas[0] + ZPE_gas[0] - TS_gas[0] + E_solv_gas[0]
        G_H2Ol = E_DFT_gas[1] + ZPE_gas[1] - TS_gas[1] + E_solv_gas[1]
        G_O2g = 2 * (G_H2Ol - G_H2g) + 4 * U_0
        G_H_e = 0.5*G_H2g - U*n
        """Gibbs energies of reaction"""
        G1 = G_OOH - G_O2g - G_H_e
        G2 = G_Ofcc + G_H2Ol - G_OOH - G_H_e
        G2a = G_Oatop + G_H2Ol - G_OOH - G_H_e
        G2b = G_Ofcc + G_OH - G_OOH
        G3 = G_OH - G_Ofcc - G_H_e
        G3a = G_OH - G_Oatop - G_H_e
        G4 = G_H2Ol - G_OH - G_H_e
        G_O2fcc = 2*G_Ofcc - G_O2g
        """computing rate constants"""  
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
        """computing rates"""
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
        """changes in coverage"""
        dThetaOOHdt = r1 - r_1 - r2 + r_2 - r2a + r_2a - r2b + r_2b
        dThetaOHdt = r2b - r_2b + r3 - r_3 + r3a - r_3a - r4 + r_4
        dThetaOfccdt = rOfcc - r_Ofcc + r2 - r_2 + r2b - r_2b - r3 + r_3 
        dThetaOatopdt = r2a - r_2a - r3a + r_3a
        dydt = [dThetaOHdt,dThetaOOHdt,dThetaOfccdt,dThetaOatopdt]
        return dydt
    
    def rate_terrace(self,Theta,popt,GCN_scaling):
        kB = 8.617e-5                      # eV / K
        h = 4.135667662e-15;               # eV * s
        T = 298.15                         # K
        U_0 = 1.23                         # eV, theoretical maximum cell voltage for ORR
        U = 0.9                            # V, cathode potential
        pO2g = 1; pH2 = 1; pH2O = 1         #Pressures of O2, H2 and H2O [atm]
        hO2 = 0.0013                        #Henry's constant in mol/(kg*bar)
        kg2mol = 55.5                   #moles of water in 1 kg H2O
        pO2 = hO2*pO2g/kg2mol
        n = 1                               # number of electrons tranfered in each step    
        # *OH, *OOH, O*
        ZPE = [0.332, 0.428, 0.072]                # zero-point energy correction, eV
        TS = [0, 0, 0]                         # entropy contribution to Gibbs energy at 298 K, eV
        #Getting Coverages
        OHcov = Theta[0]; OOHcov = Theta[1]; Ocovfcc = Theta[2]; Ocovatop = Theta[3]
        #Calculating Coverage Dependent Adsorption Energies   
        dE_OH = self.dGdOH(np.array([OHcov,OOHcov,Ocovfcc]),popt,GCN_scaling[0])
        dE_Ofcc = self.dGdO(np.array([OHcov,OOHcov,Ocovfcc]),popt)
        dE_OOH = self.dGdOOH(np.array([OHcov,OOHcov,Ocovfcc]),popt,GCN_scaling[1])
        # Surface Species free energies at T = 298K
        G_OH = dE_OH + ZPE[0] - TS[0] #G minus G of surface
        G_OOH = dE_OOH + ZPE[1] - TS[1] # G minus G of surface
        G_Ofcc = dE_Ofcc + ZPE[2] - TS[2] #RPBE-PBE G_O = 0.316
        G_Oatop = G_Ofcc + -212.88971 - -214.35223
        # Gas species Gibbs energies
        # H2(g), H2O(l), O2(g), OH(g), OOH(g), O2 (g)
        E_DFT_gas = [-6.7595, -14.2222, -9.86] # From my own DFT data
        # H2, H2O(l), O2(gas)
        ZPE_gas = [0.270, 0.574, 0.0971]  # eV 
        TS_gas = [0.404, 0.583, 0.634]  # at 298 K, eV / K
        E_solv_gas = [0, -0.087]  # eV H2O(l) solvation if TS(g) at 298K
        """Computing Gibbs energies of gas and solvated species"""
        G_H2g = E_DFT_gas[0] + ZPE_gas[0] - TS_gas[0] + E_solv_gas[0]
        G_H2Ol = E_DFT_gas[1] + ZPE_gas[1] - TS_gas[1] + E_solv_gas[1]
        G_O2g = 2 * (G_H2Ol - G_H2g) + 4 * U_0
        G_H_e = 0.5*G_H2g - U*n
        """Gibbs energies of reaction"""
        G1 = G_OOH - G_O2g - G_H_e
        G2 = G_Ofcc + G_H2Ol - G_OOH - G_H_e
        G2a = G_Oatop + G_H2Ol - G_OOH - G_H_e
        G2b = G_Ofcc + G_OH - G_OOH
        G3 = G_OH - G_Ofcc - G_H_e
        G3a = G_OH - G_Oatop - G_H_e
        G4 = G_H2Ol - G_OH - G_H_e
        G_O2fcc = 2*G_Ofcc - G_O2g
        """computing rate constants"""  
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
        """computing rates"""
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
        rate_electron = r1-r_1+r2-r_2+r2a-r_2a+r3-r_3+r3a-r_3a+r4-r_4
        return rate_electron
    
    def coverage_cavity_edge(self,Theta,t,popt_terrace,popt_cavity_edge,GCN_scaling_cavity,GCN_scaling_edge):
        kB = 8.617e-5                      # eV / K
        h = 4.135667662e-15;               # eV * s
        T = 298.15                         # K
        U_0 = 1.23                         # eV, theoretical maximum cell voltage for ORR
        U = 0.9                            # V, cathode potential
        pO2g = 1; pH2 = 1; pH2O = 1         #Pressures of O2, H2 and H2O [atm]
        hO2 = 0.0013                        #Henry's constant in mol/(kg*bar)
        kg2mol = 55.5                   #moles of water in 1 kg H2O
        pO2 = hO2*pO2g/kg2mol
        n = 1                               # number of electrons tranfered in each step    
        # *OH, *OOH, O*
        ZPE = [0.332, 0.428, 0.072]                # zero-point energy correction, eV
        TS = [0, 0, 0]                         # entropy contribution to Gibbs energy at 298 K, eV
        #Getting Coverages
        OHedge = Theta[0]; OHcav = Theta[1]; OOHedge = Theta[2]; OOHcav = Theta[3]
        Ocovfcc = Theta[4]; Ocovatop = Theta[5]
        #Calculating Coverage Dependent Adsorption Energies   
        dE_OHedge = self.dGdOHedge(np.array([OHedge,OHcav,OOHedge,OOHcav,(Ocovfcc+Ocovatop)]),popt_cavity_edge,popt_terrace,GCN_scaling_edge[0])
        dE_OHcav = self.dGdOHcav(np.array([OHedge,OHcav,OOHedge,OOHcav,(Ocovfcc+Ocovatop)]),popt_cavity_edge,popt_terrace,GCN_scaling_cavity[0])
        dE_OOHedge = self.dGdOOHedge(np.array([OHedge,OHcav,OOHedge,OOHcav,(Ocovfcc+Ocovatop)]),popt_cavity_edge,popt_terrace,GCN_scaling_edge[1])
        dE_OOHcav = self.dGdOOHcav(np.array([OHedge,OHcav,OOHedge,OOHcav,(Ocovfcc+Ocovatop)]),popt_cavity_edge,popt_terrace,GCN_scaling_cavity[1])
        dE_Ofcc = self.dGdO(np.array([(OHedge+OHcav),(OOHedge+OOHcav),(Ocovfcc+Ocovatop)]),popt_terrace)
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
        """Rate constants and activation energies"""
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
        """rates"""
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
        """changes in coverage"""
        dThetaOOHedgedt = r1edge - r_1edge - r2edge + r_2edge - r2aedge + r_2aedge - r2bedge + r_2bedge
        dThetaOHedgedt = r2bedge - r_2bedge + r3edge - r_3edge + r3aedge - r_3aedge - r4edge + r_4edge #+ r3b - r_3b
        dThetaOOHcavdt = r1cav - r_1cav - r2cav + r_2cav - r2acav + r_2acav - r2bcav + r_2bcav
        dThetaOHcavdt = r2bcav - r_2bcav + r3cav - r_3cav + r3acav - r_3acav - r4cav + r_4cav #+ r3b - r_3b
        r2 = r2edge+r2cav; r_2 = r_2edge+r_2cav; r2a = r2aedge+r2acav; r_2a = r_2aedge+r_2acav
        r2b = r2bedge+r2bcav; r_2b = r_2bedge+r_2bcav; r3=r3edge+r3cav; r_3=r_3edge+r_3cav; 
        r3a = r3aedge+r3acav; r_3a=r_3aedge+r_3acav
        dThetaOfccdt = rOfcc - r_Ofcc + r2 - r_2 + r2b - r_2b - r3 + r_3 
        dThetaOatopdt = r2a - r_2a - r3a + r_3a
        dydt = [dThetaOHedgedt,dThetaOHcavdt,dThetaOOHedgedt,dThetaOOHcavdt,dThetaOfccdt,dThetaOatopdt]
        return dydt
    
    def rate_cavity_edge(self,Theta,popt_terrace,popt_cavity_edge,GCN_scaling_cavity,GCN_scaling_edge):
        kB = 8.617e-5                      # eV / K
        h = 4.135667662e-15;               # eV * s
        T = 298.15                         # K
        U_0 = 1.23                         # eV, theoretical maximum cell voltage for ORR
        U = 0.9                            # V, cathode potential
        pO2g = 1; pH2 = 1; pH2O = 1         #Pressures of O2, H2 and H2O [atm]
        hO2 = 0.0013                        #Henry's constant in mol/(kg*bar)
        kg2mol = 55.5                   #moles of water in 1 kg H2O
        pO2 = hO2*pO2g/kg2mol
        n = 1                               # number of electrons tranfered in each step    
        # *OH, *OOH, O*
        ZPE = [0.332, 0.428, 0.072]                # zero-point energy correction, eV
        TS = [0, 0, 0]                         # entropy contribution to Gibbs energy at 298 K, eV
        #Getting Coverages
        OHedge = Theta[0]; OHcav = Theta[1]; OOHedge = Theta[2]; OOHcav = Theta[3]
        Ocovfcc = Theta[4]; Ocovatop = Theta[5]
        #Calculating Coverage Dependent Adsorption Energies   
        dE_OHedge = self.dGdOHedge(np.array([OHedge,OHcav,OOHedge,OOHcav,(Ocovfcc+Ocovatop)]),popt_cavity_edge,popt_terrace,GCN_scaling_edge[0])
        dE_OHcav = self.dGdOHcav(np.array([OHedge,OHcav,OOHedge,OOHcav,(Ocovfcc+Ocovatop)]),popt_cavity_edge,popt_terrace,GCN_scaling_cavity[0])
        dE_OOHedge = self.dGdOOHedge(np.array([OHedge,OHcav,OOHedge,OOHcav,(Ocovfcc+Ocovatop)]),popt_cavity_edge,popt_terrace,GCN_scaling_edge[1])
        dE_OOHcav = self.dGdOOHcav(np.array([OHedge,OHcav,OOHedge,OOHcav,(Ocovfcc+Ocovatop)]),popt_cavity_edge,popt_terrace,GCN_scaling_cavity[1])
        dE_Ofcc = self.dGdO(np.array([(OHedge+OHcav),(OOHedge+OOHcav),(Ocovfcc+Ocovatop)]),popt_terrace)
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
        """Rate constants and activation energies"""
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
        """rates"""
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
        rate_electron_edge = r1edge-r_1edge+r2edge-r_2edge+r2aedge-r_2aedge
        +r3edge-r_3edge+r3aedge-r_3aedge+r4edge-r_4edge
        rate_electron_cavity = r1cav-r_1cav+r2cav-r_2cav+r2acav-r_2acav
        +r3cav-r_3cav+r3acav-r_3acav+r4cav-r_4cav
        return rate_electron_cavity,rate_electron_edge
    
    def get_coverage(self,GCN):
        if self.site_type == 'terrace':
            E_gas = np.array([-7.73, -13.25]) #OH and OOH
            x = metal('Pt')
            BEs_zerocov = x.get_BEs(GCN, uncertainty = False, correlations = False) #binding energies at 0 coverage for OH and OOH, respecitively, without solvation effects
            BEs_7_5 = x.get_BEs(7.5, uncertainty = False, correlations = False)
            GCN_scaling =  (BEs_zerocov + E_gas)/(BEs_7_5+E_gas)
            n = range(2,5)
            m = range(-8,6)
            for i in n:
                for ii in m:
                    t = np.linspace(0, 10**ii, 10**i)
                    sol = odeint(self.coverage_terrace, [6.14313809e-06, 3.56958665e-12, 1.93164910e-01, 7.73636912e-12]
                    , t, args=(self.popt,GCN_scaling))
                    diffm =  np.abs(sol[-4:-1].ravel() - sol[-3:].ravel())
                    if max(diffm) < 10**-12:
                        break
                diffn = np.abs(sol[1:].ravel()-sol[0:-1].ravel())
                if max(diffn) < 0.5:
                        break
        if self.site_type =='cavity_edge':
            E_gas = np.array([-7.73, -13.25]) #OH and OOH
            x = metal('Pt')
            BEs_cavity = x.get_BEs(GCN[0], uncertainty = False, correlations = False) #binding energies at 0 coverage for OH and OOH, respecitively, without solvation effects
            BEs_edge = x.get_BEs(GCN[1], uncertainty = False, correlations = False) #binding energies at 0 coverage for OH and OOH, respecitively, without solvation effects
            BEs_8_5 = x.get_BEs(8.5, uncertainty = False, correlations = False)
            BEs_5_167 = x.get_BEs(5.167, uncertainty = False, correlations = False)
            GCN_scaling_cavity =  (BEs_cavity + E_gas)/(BEs_8_5+E_gas)
            GCN_scaling_edge =  (BEs_edge + E_gas)/(BEs_5_167+E_gas)
            n = range(2,5)
            m = range(-7,6)
            for i in n:
                for ii in m:
                    t = np.linspace(0, 10**ii, 10**i)
                    sol = odeint(self.coverage_cavity_edge, [3.92747564e-01, 4.69466971e-04,   7.72294626e-07,
          8.80151673e-13,   3.96037416e-04,   1.37359967e-11]
                    , t, args=(self.popt,self.popt_cavity_edge,GCN_scaling_cavity,GCN_scaling_edge))
                    diffm =  np.abs(sol[-4:-1].ravel() - sol[-3:].ravel())
                    if max(diffm) < 10**-12:
                        break
                diffn = np.abs(sol[1:].ravel()-sol[0:-1].ravel())
                if max(diffn) < 0.2:
                        break
        return t, sol
    
    def get_rate(self,GCN,coverage):
        if self.site_type == 'terrace':
            E_gas = np.array([-7.73, -13.25]) #OH and OOH
            x = metal('Pt')
            BEs_zerocov = x.get_BEs(GCN, uncertainty = False, correlations = False) #binding energies at 0 coverage for OH and OOH, respecitively, without solvation effects
            BEs_7_5 = x.get_BEs(7.5, uncertainty = False, correlations = False)
            GCN_scaling =  (BEs_zerocov + E_gas)/(BEs_7_5+E_gas)
            sol = odeint(self.coverage_terrace, coverage
                    , np.linspace(0, 1, 10**6), args=(self.popt,GCN_scaling))
            rate = self.rate_terrace(sol[-1],self.popt,GCN_scaling)
            return(rate)
        if self.site_type == 'cavity_edge':
            E_gas = np.array([-7.73, -13.25]) #OH and OOH
            x = metal('Pt')
            BEs_cavity = x.get_BEs(GCN[0], uncertainty = False, correlations = False) #binding energies at 0 coverage for OH and OOH, respecitively, without solvation effects
            BEs_edge = x.get_BEs(GCN[1], uncertainty = False, correlations = False) #binding energies at 0 coverage for OH and OOH, respecitively, without solvation effects
            BEs_8_5 = x.get_BEs(8.5, uncertainty = False, correlations = False)
            BEs_5_167 = x.get_BEs(5.167, uncertainty = False, correlations = False)
            GCN_scaling_cavity =  (BEs_cavity + E_gas)/(BEs_8_5+E_gas)
            GCN_scaling_edge =  (BEs_edge + E_gas)/(BEs_5_167+E_gas)
            sol = odeint(self.coverage_cavity_edge, coverage
                    , np.linspace(0, 1, 10**6), args=(self.popt,self.popt_cavity_edge,GCN_scaling_cavity,GCN_scaling_edge))
            rate = self.rate_cavity_edge(sol[-1],self.popt,self.popt_cavity_edge,GCN_scaling_cavity,GCN_scaling_edge)
            return rate
            