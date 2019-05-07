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
#fsum provides better summation accuracy
from math import fsum

import sys
from orr_optimizer.metal import metal


class ORR_MKM:
    """
    Class for implementing the oxygen reduction reaction (ORR) microkinetic model (MKM).
    
    The MKM accounts for coverage and explicit liquid water solvation effects.
    
    The MKM uses scipy's odeint to find the steady state surface coverages of
    atomic oxygen (O), hydroxyl (OH), and hydroperoxyl (OOH) at the specified
    generalized coordination number (GCN) using get_coverage().
    The intensive rate at all relevant GCNs can also be obtained using get_rate().

    Input
        site_type: string
            The site type of the desired coverages and rates. It can either be
            terrace (Pt111 without defects)

    Interactive Methods
        get_coverage(GCN)
            Obtain steady state coverage at a specified GCN [ML]
        get_rate(GCN,coverage)
            Obtain rate at the specified GCN and coverage [mA/atom]
    
    Internal Methods
        coverage(GCN_scaling)
            Implements the relevant coverage method in an ode solver
        coveragefunc(Theta,t,popt,GCN_scaling,GET_RATE=False)
            Returns coverages solved through ode integrations or rate at either
            terrace or edge sites.
        coverage_cavity_edge(Theta,t,popt_terrace,poptO, popt_cavity_edge
        ,GCN_scaling_cavity,GCN_scaling_edge,GET_RATE=False)
            Returns coverages solve through ode integrations or rate at 
            edge and cavity sites that are coupled.
        gcn_scaling(GCN,GCN_reference)
            Amount to shift derivatives of surface energy due to differences
            in GCN between the MKM and the DFT used to parameterize the
            Hamiltonian
        Gfit()
            Fits parameters of the Hamiltonian for terrace and edge sites
        Gfit_cavity_edge()
            Fits parameters for Hamiltonian where edge and cavity sites are 
            coupled and therefore both considered in the same DFT calculations.
        rate(coverage,GCN_scaling)
            Implements the relevant rate method in an ode solver
    """
    def __init__(self, site_type):
        """
        Attributes Generated
            site_type: string
                Directs which DFT Data to be used, and the MKM to run
            G_H2Osurf: scalar
                Water replacment energy when displaced by OH or OOH
        """
        self.site_type = site_type
        E_H2Og = -14.219432 # water in vacuum
        E7H2O = -379.78779 # water in cavity
        E6H2O = -365.04325 # removing H2O from cavity
        Esolv_H2O_explicit = E7H2O-E6H2O-E_H2Og #this is the energy of solvated H2O interacting with a surface
        self.G_H2Osurf = E_H2Og + Esolv_H2O_explicit #This is used to calculate the water replacement energy
        self.Gfit()
        self.Gfit_cavity_edge()

    def Gfit(self):
        """
        Gfit loads either Pt terrace (GCN=7.5) or 6.4 GCN edge data and uses it
        to parameterize the Hamiltonian. It also generates functional attributes
        for calculating the derivatives of surface energy with respect to
        OH, OOH and O.
        
        Attributes used
            site_type: string
                Terrace, Edge, or cavity_edge
            G_H2Osurf: scalar
                Energy of solvated H2O interacting with a surface
        
        Attributes Generated
            Derivatives of the Hamiltonian with respect to each surface species
            The following determine changes in surface energy energy given 
            coverage, a set of parameters used to fit a Hamiltonian, and a
            shift in zero coverage energy determined by the GCN used.
            popt: array of length 9
                Contains Hamiltonian fitted parameters for the terrace
                and edge (6.4 GCN) DFT data
            poptO: array of length 8
                Contains parameters of Hamiltonian fit to 6.4 edge GCN data but
                used in determining the repulsive effects of oxygen on OH (edge),
                OOH (edge), OH (cavity), and OOH (cavity) in the microkinetic
                model for coupled edge and cavity sites.
            dGdOH: function
                Change in surface energy due to binding of OH 
            dGdOOH: function
                Change in surface energy due to binding of OOH
            dGdO: function
                Change in surface energy due to binding of of O
        """
        data_file = ''
        #Go is the energy of the bare slab with 12 water molecules (2 full layers)
        Go = 0
        if self.site_type == 'terrace':
        #Surface energies on Pt111 without defects.
            data_file = 'Surface_Energies.csv'
            Go = -385.40342
#==============================================================================
#       if the site type is cavity_edge, the oxygen adsorption energy
#       for the MKM is determined using the parameters fit to the 6.4 edge GCN data
#       The 6.4 edge GCN edge has no cavity so the MKM for the undefected
#       surface can be used with different parameters.
#==============================================================================
        if self.site_type=='edge' or self.site_type =='cavity_edge':
            data_file = 'Surface_Energies_6_4.csv'
            Go = -378.28072
        data_file = os.path.expanduser(data_file)
        CovDat = read_csv(data_file)
#==============================================================================
#       Coverages contains all O, OH, and OOH coverages used in regressing
#       the Hamiltonian
#==============================================================================

        Coverages = np.array([CovDat.OHcov,CovDat.OOHcov,CovDat.Ocov])
#==============================================================================
#       WaterReplacement is the total energy of the water molecules that is
#       not accounted for in the DFT calculation because they are replaced
#       by the OH, or OOH adsorbates. O is in an fcc site so the number of
#       water molecules in the DFT calculations are not affected by its 
#       presence.
#==============================================================================
        WaterReplacement = np.sum(CovDat[['OHcov','OOHcov']],axis=1)*9*self.G_H2Osurf

        #Gsurf is the Hamiltonian. It is the surface energy with adsorbates
        def Gsurf(Coverageinput,s,tp,u,x,y,z,GOHo,GOOHo,GOo):
            OHcov, OOHcov, Ocov = Coverageinput
            Gval = (GOHo*OHcov + GOOHo*OOHcov + GOo*Ocov
                    + s*(tp*Ocov+OHcov)**u + x*(y*Ocov+OHcov)**z*OOHcov)
            return Gval
        
#==============================================================================
#       Energies from DFT minus the surface energy of the surface with just
#       the 12 water molecules. We also add back the energy of the water
#       molecules since they are replaced by the OH/OOH in the honeycomb
#       structure based DFT calculations
#==============================================================================
        Energies = CovDat.Energy.as_matrix() + WaterReplacement - Go
        #these bounds limit the parameters in the Hamiltonian so that
        #exponents-1 and the base are not negative.
        lmin = 0
        lmax = 30
        emin = 1
        emax=4
        #nonlinear least squares fit of Hamiltonian parameters
        self.popt, pcov = curve_fit(Gsurf,Coverages,Energies/9.0
        ,bounds=(np.array([lmin,lmin,emin,lmin,lmin,emin,-20,-20,-20])
        ,np.array([lmax,lmax,emax,lmax,lmax,emax,0,0,0])))
        
        #surface for Hamiltonian parameteirzied with the 6.4 edge data but used
        #for the microkintic model for coupled edge and cavity sites.
        def GsurfO(Coverageinput,s,tp,u,xO,yO,GOHo,GOOHo,GOo):
            OHcov, OOHcov, Ocov = Coverageinput
            Gval = (GOHo*OHcov + GOOHo*OOHcov + GOo*Ocov
                    + s*(tp*OHcov+OOHcov)**u + xO*OHcov*Ocov + yO*OOHcov*Ocov)
            return Gval
        
        #these bounds limit the parameters in the Hamiltonian so that
        #exponents-1 and the base are not negative.
        lmin = 0
        lmax = 30
        emin = 1
        emax=4
        #nonlinear least squares fit of Hamiltonian parameters
        self.poptO, pcovO = curve_fit(GsurfO,Coverages,Energies/9.0
        ,bounds=(np.array([lmin,lmin,emin,lmin,lmin,-20,-20,-20])
        ,np.array([lmax,lmax,emax,lmax,lmax,0,0,0])))
#==============================================================================
#         The following functions take in a coverage, values for regressed
#         Hamiltonian parameter, and a value to adust the zero coverage surface
#         energy due to changes in GCN. The output is the change in surface
#         energy of the relevent species at the inputted coverages
#==============================================================================
        def dGdOH(Coverageinput,popt,GCN_scaling):
            """
            Calculates the derivative of surface energy with respect to OH
            on a terrace or edge site
            
            Inputs
                Coverageinput: length 3 array
                    Coverages of OH, OOH, and O
                popt: length 9 array
                    Parameters of fitted Hamiltonian
                GCN_scaling: scalar
                    Value to shift zero coveage surface energy change due to 
                    GCN of the site being different than DFT data
            Output
                dGval: scalar
                    Change in surface energy due to adsorption of an OH molecule (eV/molecule)
            """
            s,tp,u,x,y,z,GOHo,GOOHo,GOo = popt
            #set negative coverages from numerical error of ode solver to 0
            Coverageinput = [i if i>0 else 0 for i in Coverageinput]
            OHcov, OOHcov, Ocov = Coverageinput
            dGval = (GOHo+GCN_scaling + u*s*(tp*Ocov+OHcov)**(u-1)
            + z*x*(y*Ocov+OHcov)**(z-1)*OOHcov)
            return dGval          
        def dGdOOH(Coverageinput,popt,GCN_scaling):
            """
            Calculates the derivative of surface energy with respect to OOH
            on a terrace or edge site
            
            Inputs
                Coverageinput: length 3 array
                    Coverages of OH, OOH, and O
                popt: length 9 array
                    Parameters of fitted Hamiltonian
                GCN_scaling: scalar
                    Value to shift zero coverage zero coverage energy change
                    due to GCN of the site being different than DFT data
            Output
                dGval: scalar
                    Change in surface energy due to adsorption of an OOH molecule (eV/molecule)
            """
            s,tp,u,x,y,z,GOHo,GOOHo,GOo = popt
            #set negative coverages from numerical error of ode solver to 0
            Coverageinput = [i if i>0 else 0 for i in Coverageinput] 
            OHcov, OOHcov, Ocov = Coverageinput
            dGval = GOOHo+GCN_scaling + x*(y*Ocov+OHcov)**z
            return dGval

        def dGdO(Coverageinput,popt,GCN_scaling):
            """
            Calculates the derivative of surface energy with respect to O
            on a terrace/cavity site or a coupled edges and cavities
            Note different input shapes if site_type=='cavity_edge'
            
            Inputs
                Coverageinput: length 3 array or array of shape (3,2)
                    Coverages of OH, OOH, and O or
                    [OHedge, OHcav], [OOHedge, OOHcav] and [Oedge, Ocav]
                popt: length 9 array
                    Parameters of fitted Hamiltonian
                GCN_scaling: scalar or length two array
                    Value to shift zero coverage surface energy change due to GCN of
                    of site being different than DFT data
            Output
                dGval: scalar or length 2 array (if site_type=='cavity edge')
                    Change in surface energy due to adsorption of an O atom (eV/atom)
                    if site_type=='cavity edge' it is the change in surface
                    energy due to O adsorption on a edge and cavity site, respectively
            """
            s,tp,u,x,y,z,GOHo,GOOHo,GOo = popt
            #dGval is an array of length 2 (for cavity and edge sites combined)
            #dGval[0] is for the edge site and dGval[1] is for the cavity site
            if self.site_type == 'cavity_edge':
#==============================================================================
#                 -6.46 eV is the oxygen adsorption energy on 6.4 GCN edge without 
#                 adsorbates. Used to correct zero coverage enregy for 
#                 oxygen on 8.5 GCN cavity and 5.1 GCN edge for which DFT 
#                 calculations with oxygen (with other adsorbates) were not performed.
#                 -6.57 and -5.12 are the O adsorption energies on the 5.1 eV 
#==============================================================================
                GOo = np.array([GOo,GOo])+np.array([-6.57278+6.46064,-5.12679+6.46064]) 
                #set negative coverages from numerical error of ode solver to 0
                Coverageinput = [np.array([i if i>0 else 0 for i in Coverageinput[0]])
                ,np.array([i if i>0 else 0 for i in Coverageinput[1]])
                ,np.array([i if i>0 else 0 for i in Coverageinput[2]])]
            else:
                #set negative coverages from numerical error of ode solver to 0
                Coverageinput = [i if i>0 else 0 for i in Coverageinput]
            OHcov, OOHcov, Ocov = Coverageinput
            dGval = (GOo+GCN_scaling + tp*u*s*(tp*Ocov+OHcov)**(u-1)
                    +y*z*x*(y*Ocov+OHcov)**(z-1)*OOHcov)
            return dGval
        #set method attributes to surface energy derivative functions so they 
        #can be used by other methods in the orr_mkm.py class
        self.dGdOH = dGdOH
        self.dGdOOH = dGdOOH
        self.dGdO = dGdO
        
        #for Analysis use strictly outside the MKM
        self.Gsurf = Gsurf
        self.DFT_ENERGIES = Energies
        self.GsurfO = GsurfO
        self.DFT_COVERAGES = Coverages
        
    def Gfit_cavity_edge(self):
        """
        Gfit_cavity_edge loads Pt DFT data for the coupled 5.1 GCN edge sites 
        and 8.5 GCN cavity site the hamiltonian. It also generates functional
        attributes for calculating the derivative of surface energy with 
        respect to  OH, OOH and O.
        
        Attributes used:
            G_H2Osurf: scalar
                Energy of solvated H2O interacting with a surface
        
        Attributes Generated
            Derivatives of the Hamiltonian with respect to each surface species
            dGdOHedge: function
                Change in surface energy from adsorption of OH on an edge site
            dGdOOHedge: function
                Change in surface energy from adsorption of OOH on an edge site
            dGdOHcav: function
                Change in surface energy from adsorption OH on an cavity site
            dGdOOHcav: function
                Change in surface energy from adsorption OOH on an cavity site
        """
        #DFT Data for edge sites with a GCN of 5.167 and a single 8.5 GCN cavity site
        data_file = 'Surface_Energies_cavity.csv'
        data_file = os.path.expanduser(data_file)
        CovDat = read_csv(data_file)
        #There are two sets of coverages for adosrbed OH and OOH, one at the 
        #edge and one at the cavity
        Coverages = np.array([CovDat.OH_edge,CovDat.OH_cavity,CovDat.OOH_edge
                              ,CovDat.OOH_cavity])
        #Energy to of water molecules interacting with a surface and replaced
        #in the honeycome strucutre by OH and OOH
        WaterReplacement = np.sum(Coverages,axis=0)*9*self.G_H2Osurf
        #Hamiltonian for Energy of a cavity with an edge site.
        def Gsurf(Coverageinput,x,x2,x3,y,z,GOHedgeo,GOHcavo,GOOHedgeo,GOOHcavo):
            OHedge, OHcav, OOHedge, OOHcav = Coverageinput
            Gval = (GOHedgeo*OHedge + GOHcavo*OHcav + GOOHedgeo*OOHedge
            + GOOHcavo*OOHcav + x*(y*OHedge+OOHedge)**z
            + x2*(OHedge+OOHedge)*OHcav + x3*(OHedge+OOHedge)*OOHcav)
            return Gval
        
        #Go is the energy of the bare slab with 12 water molecules (2 full layers)
        Go = -365.04325
        #Energies used to fit surface energy Hamiltonian
        Energies = CovDat.Energy.as_matrix() + WaterReplacement - Go
        #these bounds limit the parameters in the Hamiltonian so that
        #exponents-1 and the base are not negative.
        lmin = 0
        lmax = 30
        emin = 1
        emax=4
        self.popt_cavity_edge, pcov = curve_fit(Gsurf,Coverages,Energies/9.0
            ,bounds=(np.array([lmin,lmin,lmin,lmin,emin,-20,-20,-20,-20])
            ,np.array([lmax,lmax,lmax,lmax,emax,0,0,0,0])))
        def dGdOHedge(Coverageinput,popt,poptO,GCN_scaling):
            """
            Calculates the derivative of surface energy with respect to OH on an edge site
            
            Inputs
                Coverageinput: length 5 array
                    Coverages of OH (edge), OH (cavity)
                    , OOH (edge), OOH (cavity) and O
                popt: length 9 array
                    Parameters of fitted Hamiltonian for coupled edges/cavity
                poptO: length 8 array
                    Parameters fitted to the Hamiltonian for the terrace/edge 
                    site without cavities for determining repuslive effects of 
                    adsorbed oxygen. Uses DFT data for 6.4 GCN edge site
                GCN_scaling: scalar
                    Value to shift zero coveage surface energy change due to GCN of
                    of site being different than DFT data
            Output
                dGval: scalar
                    Change in surface energy due to adsorption of an OH on an
                    edge site (eV/molecule)
            """
            x,x2,x3,y,z,GOHedgeo,GOHcavo,GOOHedgeo,GOOHcavo = popt
            (s,tp,u,xO,yO,GOHo,GOOHo,GOo) = poptO
            #set negative coverages from numerical error of ode solver to 0
            Coverageinput = [i if i>0 else 0 for i in Coverageinput]
            OHedge, OHcav, OOHedge, OOHcav, Ocov = Coverageinput
#==============================================================================
#             x*y*z*y**(z-1)/(s*tp*u*tp**(u-1)) is used to correct the value of xO by the ratio of
#             OH edge repulsive effects in coupled edge cavity site Hamiltonian
#             over the OH repulsive effects in the edge Hamiltonian
#==============================================================================
            dGval = (GOHedgeo+GCN_scaling + y*x*z*(y*OHedge+OOHedge)**(z-1) 
            + x2*OHcav + x3*OOHcav + x*z*y**z/(s*u*tp**u)*xO*Ocov)
            return dGval

        def dGdOHcav(Coverageinput,popt,poptO,GCN_scaling):
            """
            Calculates the derivative of surface energy with respect to OH on 
            a cavity site
            
            Inputs
                Coverageinput: length 5 array
                    Coverages of OH (edge), OH (cavity)
                    , OOH (edge), OOH (cavity) and O
                popt: length 9 array
                    Parameters of fitted Hamiltonian for coupled edges/cavity
                poptO: length 8 array
                    Parameters fitted to the Hamiltonian for the terrace/edge 
                    site without cavities for determining repuslive effects of 
                    adsorbed oxygen. Uses DFT dat afor 6.4 GCN edge site
                GCN_scaling: scalar
                    Value to shift zero coverage surface energy derivative due to GCN of
                    of site being different than DFT data
            Output
                dGval: scalar
                    Change in surface energy due to adsorption of an OH on a
                    cavity site (eV/molecule)
            """
            x,x2,x3,y,z,GOHedgeo,GOHcavo,GOOHedgeo,GOOHcavo = popt
            (s,tp,u,xO,yO,GOHo,GOOHo,GOo) = poptO
            #set negative coverages from numerical error of ode solver to 0
            Coverageinput = [i if i>0 else 0 for i in Coverageinput]
            OHedge, OHcav, OOHedge, OOHcav, Ocov = Coverageinput
#==============================================================================
#             x*z*y**z/(s*tp*u*tp**(u-1)) is used to correct the value of xO by the ratio of
#             OH cavity repulsive effects in coupled edge cavity site Hamiltonian
#             over the OH repulsive effects in the edge Hamiltonian
#==============================================================================
            dGval = (GOHcavo+GCN_scaling + x2*(OHedge+OOHedge) 
            + x*z*y**z/(s*u*tp**u)*xO*Ocov)
            return dGval
        
        def dGdOOHedge(Coverageinput,popt,poptO,GCN_scaling):
            """
            Calculates the derivative of surface energy with respect to OOH
            on an edge site
            
            Inputs
                Coverageinput: length 5 array
                    Coverages of OH (edge), OH (cavity)
                    , OOH (edge), OOH (cavity) and O
                popt: length 9 array
                    Parameters of fitted Hamiltonian for coupled edges/cavity
                poptO: length 8 array
                    Parameters fitted to the Hamiltonian for the terrace/edge 
                    site without cavities for determining repuslive effects of 
                    adsorbed oxygen. Uses DFT dat afor 6.4 GCN edge site
                GCN_scaling: scalar
                    Value to shift zero coverage surface energy derivative due to GCN of
                    of site being different than DFT data
            Output
                dGval: scalar
                    Change in surface energy due to adsorption of an OOH on an
                    edge site (eV/molecule)
            """
            x,x2,x3,y,z,GOHedgeo,GOHcavo,GOOHedgeo,GOOHcavo = popt
            (s,tp,u,xO,yO,GOHo,GOOHo,GOo) = poptO
            #set negative coverages from numerical error of ode solver to 0
            Coverageinput = [i if i>0 else 0 for i in Coverageinput]
            OHedge, OHcav, OOHedge, OOHcav, Ocov = Coverageinput
#==============================================================================
#             x*z*y**(z-1)/(s*u*tp**(u-1)) is used to correct the value of yO by the ratio of
#             OOH edge repulsive effects in coupled edge cavity site Hamiltonian
#             over the OOH repulsive effects in the edge Hamiltonian
#==============================================================================
            dGval = (GOOHedgeo+GCN_scaling + x*z*(y*OHedge+OOHedge)**(z-1)
            + x2*OHcav + x3*OOHcav + x*z*y**(z-1)/(s*u*tp**(u-1))*yO*Ocov)
            return dGval
        
        def dGdOOHcav(Coverageinput,popt,poptO,GCN_scaling):
            """
            Calculates the derivative of surface energy with respect to OOH
            on an cavity site
            
            Inputs
                Coverageinput: length 5 array
                    Coverages of OH (edge), OH (cavity)
                    , OOH (edge), OOH (cavity) and O
                popt: length 9 array
                    Parameters of fitted Hamiltonian for coupled edges/cavity
                poptO: length 8 array
                    Parameters fitted to the Hamiltonian for the terrace/edge 
                    site without cavities for determining repuslive effects of 
                    adsorbed oxygen. Uses DFT dat afor 6.4 GCN edge site
                GCN_scaling: scalar
                    Value to shift zero coveage surface energy change due to GCN of
                    of site being different than DFT data
            Output
                dGval: scalar
                    Change in surface energy due to adsorption of an OOH on a
                    cavity site (eV/molecule)
            """
            x,x2,x3,y,z,GOHedgeo,GOHcavo,GOOHedgeo,GOOHcavo = popt
            (s,tp,u,xO,yO,GOHo,GOOHo,GOo) = poptO
            #set negative coverages from numerical error of ode solver to 0
            Coverageinput = [i if i>0 else 0 for i in Coverageinput]
            OHedge, OHcav, OOHedge, OOHcav,Ocov = Coverageinput
#==============================================================================
#             x*z*y**(z-1)/(s*u*tp**(u-1)) is used to correct the value of yO by the ratio of
#             OOH cavity repulsive effects in coupled edge cavity site Hamiltonian
#             over the OOH repulsive effects in the edge Hamiltonian
#==============================================================================
            dGval = (GOOHcavo+GCN_scaling + x3*(OHedge + OOHedge) 
            + x*z*y**(z-1)/(s*u*tp**(u-1))*yO*Ocov)
            return dGval
        self.dGdOHedge = dGdOHedge
        self.dGdOHcav = dGdOHcav
        self.dGdOOHedge = dGdOOHedge
        self.dGdOOHcav = dGdOOHcav
        
        #Strictly for Analysis outside the use of this MKM
        self.Gsurf_CAVEDGE = Gsurf
        self.DFT_ENERGIES_CAVEDGE = Energies
        self.DFT_COVERAGES_CAVEDGE = Coverages
    
    def coveragefunc(self,Theta,t,popt,GCN_scaling,GET_RATE=False):
        """
        Calcluates change in coverages (GET_RATE==False) or rates (GET_RATE==True)
        for the Pt(111) terrace and edge sites.
        
        Input:
            Theta: array of length 4 
                Coverages (OH, OOH, O (fcc) and O (atop)).
                These are initial guesses if GET_RATE==False
            t: 1d array
                Time steps for ode integration. Dummy variable used if GET_RATE==True
            popt: array of length 9 
                Parameters for energy derivative functions determined from regressing the Hamiltonian
            GCN_scaling: array of length 3
                Amount to shift zero coverage binding energies of adsorbates based on GCN
            GET_RATE: boolean
                Determines weather change in coverage or rate is returned
            
        Output:
            rate_electron: scalar
                Summed rate of all electrochemical steps (current) (for GET_RATE == True)
            dydt: array of length 4
                Change in coverage with the internal timestep (for GET_RATE == False)
        
        Attributes used:
            dGdOH: function
                Change in surface energy due to binding of OH 
            dGdOOH: function
                Change in surface energy due to binding of OOH
            dGdO: function
                Change in surface energy due to binding of of O
        """
        kB = 8.617e-5                      # Boltzman constant eV / K
        h = 4.135667662e-15;               # Planks constant eV * s
        T = 298.15                         # K
        U_0 = 1.23                         # eV, theoretical maximum cell voltage for ORR
        U = 0.9                            # V, cathode potential
        #pressure of H2 needs to be 1 atm as the reference is the standard hydrogen
        #electrode (SHE)
        pO2g = 1; pH2 = 1; pH2O = 1         #Pressures of O2, H2 and H2O [atm]
        hO2 = 0.0013                        #Henry's constant in mol/(kg*bar)
        kg2mol = 55.5                   #moles of water in 1 kg H2O
        pO2 = hO2*pO2g/kg2mol
        n = 1                               # number of electrons tranfered in each step    
        # *OH, *OOH, O*
        ZPE = [0.332, 0.428, 0.072]             # zero-point energy correction, eV
        TS = [0, 0, 0]                         # entropy contribution to Gibbs energy at 298 K, eV
        #Getting Coverages
        OHcov = Theta[0]; OOHcov = Theta[1]; Ocovfcc = Theta[2]; Ocovatop = Theta[3]
        #Calculating Coverage Dependent Adsorption Energies   
        dE_OH = self.dGdOH(np.array([OHcov,OOHcov,Ocovfcc]),popt,GCN_scaling[0])
        dE_Ofcc = self.dGdO(np.array([OHcov,OOHcov,Ocovfcc]),popt,GCN_scaling[2])
        dE_OOH = self.dGdOOH(np.array([OHcov,OOHcov,Ocovfcc]),popt,GCN_scaling[1])
        # Surface Species free energies at T = 298K
        G_OH = dE_OH + ZPE[0] - TS[0] #G minus G of surface
        G_OOH = dE_OOH + ZPE[1] - TS[1] # G minus G of surface
        G_Ofcc = dE_Ofcc + ZPE[2] - TS[2] #RPBE-PBE G_O = 0.316
#==============================================================================
#         Energy of oxygen on the top site is taken to be the regressed energy of
#         oxygen on the fcc site plus the difference of the atop and fcc site energy in vacuum.
#         this is because the oxygen on an atop site with explicit water molecules 
#         was very unstable and a local minima was difficult to find.
#==============================================================================
        G_Oatop = G_Ofcc + -212.88971 - -214.35223
        # Gas species Gibbs energies
        # H2(g), H2O(l), O2(g), OH(g), OOH(g), O2 (g)
        E_DFT_gas = [-6.7595, -14.2222] # From my own DFT data
        # H2, H2O(l)
        ZPE_gas = [0.270, 0.574]  # eV 
        TS_gas = [0.404, 0.583]  # at 298 K, eV / K
        E_solv_gas = [0, -0.087]  # eV H2O(l) solvation if using TS(g) at 298K
        #Computing Gibbs energies of gas and solvated species
        G_H2g = E_DFT_gas[0] + ZPE_gas[0] - TS_gas[0] + E_solv_gas[0]
        G_H2Ol = E_DFT_gas[1] + ZPE_gas[1] - TS_gas[1] + E_solv_gas[1]
        G_O2g = 2 * (G_H2Ol - G_H2g) + 4 * U_0
        G_H_e = 0.5*G_H2g - U*n
        #Gibbs energies of reaction
        #formation of OOH* by O2(gas), H+ and e-
        G1 = G_OOH - G_O2g - G_H_e
        #formation of O*(fcc) and H2O(l) by OOH*, H+ and e-
        G2 = G_Ofcc + G_H2Ol - G_OOH - G_H_e
        #formation of O*(atop) and H2O(l) by OOH*, H+ and e-
        G2a = G_Oatop + G_H2Ol - G_OOH - G_H_e
        #formation of O*(fcc) and OH* by OOH* dissociation
        G2b = G_Ofcc + G_OH - G_OOH
        #formation of OH* by O*(fcc), H+ and e-
        G3 = G_OH - G_Ofcc - G_H_e
        #formation of OH* by O*(atop), H+, and e-
        G3a = G_OH - G_Oatop - G_H_e
        #formation of H2O(l)  by OH*, H+ and e-
        G4 = G_H2Ol - G_OH - G_H_e
        #formation of 2 O*(fcc) from 1 O2(g) by dissociation
        G_O2fcc = 2*G_Ofcc - G_O2g
        #Computing rate constants
        #activation energys (Ea), forward rate constants (k) and reverse rate
        #constants (k_) correspond to the numbered reaction steps above
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
        #Computing rates
        #forward rates (r) and reverse rates (r_) correspond to the numbered 
        #rate constants and reactions above
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
        if GET_RATE == True:
            #The sum of all electrochemical steps results in the overall rate
            rate_electron = fsum([r1,-r_1,r2,-r_2,r2a,-r_2a,r3
                                  ,-r_3,r3a,-r_3a,r4,-r_4])
            return rate_electron
        else:
            #Changes in coverage
            dThetaOOHdt = fsum([r1,-r_1,-r2,r_2,-r2a,r_2a,-r2b,r_2b])
            dThetaOHdt = fsum([r2b,-r_2b,r3,-r_3,r3a,-r_3a,-r4,r_4])
            dThetaOfccdt = fsum([rOfcc,-r_Ofcc,r2,-r_2,r2b,-r_2b,-r3,r_3])
            dThetaOatopdt = fsum([r2a,-r_2a,-r3a,r_3a])
            dydt = [dThetaOHdt,dThetaOOHdt,dThetaOfccdt,dThetaOatopdt]
            return dydt
    
    def coverage_cavity_edge(self,Theta,t,popt_terrace,poptO,popt_cavity_edge
                    ,GCN_scaling_cavity,GCN_scaling_edge,GET_RATE=False):
        """
        Calcluates change in coverages (GET_RATE==False) or rates 
        (GET_RATE==True) for the coupled Pt edge and cavity sites.
        
        Input:
            Theta: array of length 8
                Coverages (OH, OOH, O (fcc) and O (atop)).
                These are initial guesses if GET_RATE==False
            t: array
                Time steps for ode integration. Dummy variable used if 
                GET_RATE==True
            popt_terrace: array of length 9
                Parameters fitted to the Hamiltonian for the terrace/edge 
                site without cavities for determining change in surface energy 
                for oxygen adsorption. Uses DFT data for 6.4 GCn edge sites.
            poptO: array of length 8
                Parameters fitted to the Hamiltonian for the terrace/edge 
                site without cavities for determining repuslive effects of 
                adsorbed oxygen. Uses DFT data afor 6.4 GCN edge sites.
            popt_cavity_edge: array of length 9
                Parameters for O surface energy derivative function based on regressing 
                the coupled edge and cavity site Hamiltonian
            GCN_scaling_cavity: array of length 3
                Amount to shift zero coverage binding energies
                of the cavity due to changes in GCN
            GCN_scaling_edge:  array of length 3
                Amount to shift zero coverage binding enregies on
                edge sites due to changes in GCN
            GET_RATE: boolean
                Determines with coverage or rate is returned
            
        Output:
            rate_electron_cavity: scalar
                Summed rate of all electrochemical steps 
                (current) (for GET_RATE == True) on the cavity
            rate_electron_edge: scalar
                Summed rate of all electrochemical steps on the
                edge sites (for GET_RATE == True)
            dydt: array of length 8
                Change in coverage with the internal timestep (for GET_RATE == False)
        
        Attributes used:
            dGdOHedge: function
                Change in surface energy from adsorption of OH on an edge site
            dGdOOHedge: function
                Change in surface energy from adsorption of OOH on an edge site
            dGdOHcav: function
                Change in surface energy from adsorption OH on an cavity site
            dGdOOHcav: function
                Change in surface energy from adsorption OOH on an cavity site
        """
        kB = 8.617e-5                   # Boltzmann constant eV / K
        h = 4.135667662e-15;            # planks constant eV * s
        T = 298.15                      # K
        U_0 = 1.23                      # eV, theoretical maximum cell voltage for ORR
        U = 0.9                         # V, cathode potential
        #pressure of H2 needs to be 1 atm as the reference is the standard hydrogen
        #electrode (SHE)
        pO2g = 1; pH2 = 1; pH2O = 1     #Pressures of O2, H2 and H2O [atm]
        hO2 = 0.0013                    #Henry's constant in mol/(kg*bar)
        kg2mol = 55.5                   #moles of water in 1 kg H2O
        pO2 = hO2*pO2g/kg2mol           #concentration of solvated O2
        n = 1                           # number of electrons tranfered in each step    
        # *OH, *OOH, O*
        ZPE = [0.332, 0.428, 0.072]     # zero-point energy correction, eV
        TS = [0, 0, 0]                  # entropy contribution to Gibbs energy at 298 K, eV
        #Getting Coverages
        OHedge = Theta[0]; OHcav = Theta[1]; OOHedge = Theta[2]; OOHcav = Theta[3]
        Ocovfccedge = Theta[4]; Ocovatopedge = Theta[5]; Ocovfcccav = Theta[6]; Ocovatopcav = Theta[7]
        #Calculating Coverage Dependent Adsorption Energies   
        dE_OHedge = self.dGdOHedge(np.array([OHedge,OHcav,OOHedge,OOHcav
                        ,(Ocovfccedge+Ocovatopedge)]),popt_cavity_edge
                        ,poptO,GCN_scaling_edge[0])
        dE_OHcav = self.dGdOHcav(np.array([OHedge,OHcav,OOHedge,OOHcav
                        ,(Ocovfcccav+Ocovatopcav)]),popt_cavity_edge
                        ,poptO,GCN_scaling_cavity[0])
        dE_OOHedge = self.dGdOOHedge(np.array([OHedge,OHcav,OOHedge,OOHcav
                        ,(Ocovfccedge+Ocovatopedge)]),popt_cavity_edge
                        ,poptO,GCN_scaling_edge[1])
        dE_OOHcav = self.dGdOOHcav(np.array([OHedge,OHcav,OOHedge,OOHcav
                        ,(Ocovfcccav+Ocovatopcav)]),popt_cavity_edge
                        ,poptO,GCN_scaling_cavity[1])
        #dE_Ofcc is a length 2 array, one value for O at the edge site and another
        #at the cavity site.
        dE_Ofcc = self.dGdO(np.array([np.array([OHedge,OHcav])
                        ,np.array([OOHedge,OOHcav])
                        ,np.array([(Ocovfccedge+Ocovatopedge)
                        ,(Ocovfcccav+Ocovatopcav)])]),popt_terrace
                        ,np.array([GCN_scaling_edge[2],GCN_scaling_cavity[2]]))
        # Species free energies at T = 298K
        G_OHedge = dE_OHedge + ZPE[0] - TS[0] #G minus G of surface
        G_OOHedge = dE_OOHedge + ZPE[1] - TS[1] # G minus G of surface
        G_OHcav = dE_OHcav + ZPE[0] - TS[0] #G minus G of surface
        G_OOHcav = dE_OOHcav + ZPE[1] - TS[1] # G minus G of surface
        #RPBE-PBE G_O = 0.316
        G_Ofcc = dE_Ofcc + ZPE[2] - TS[2]
        G_Oatop = G_Ofcc + -212.88971 - -214.35223
        # Gas species Gibbs energies
        # H2(g), H2O(l)
        E_DFT_gas = [-6.7595, -14.2222]             # From my own DFT data
        # H2, H2O(l), O2(gas)
        ZPE_gas = [0.270, 0.574]             # eV, 
        TS_gas = [0.404, 0.583]              # at 298 K, eV / K
        E_solv_gas = [0, -0.087]             # eV
        G_H2g = E_DFT_gas[0] + ZPE_gas[0] - TS_gas[0] + E_solv_gas[0]
        G_H2Ol = E_DFT_gas[1] + ZPE_gas[1] - TS_gas[1] + E_solv_gas[1]
        G_O2g = 2 * (G_H2Ol - G_H2g) + 4 * U_0
        G_H_e = 0.5*G_H2g - U*n
        #Gibbs energies of reaction
        #formation of OOH* by O2(gas), H+ and e-
        G1edge = G_OOHedge - G_O2g - G_H_e
        G1cav = G_OOHcav - G_O2g - G_H_e
        #formation of O*(fcc) and H2O(l) by OOH*, H+ and e-
        G2edge = G_Ofcc[0] + G_H2Ol - G_OOHedge - G_H_e
        G2cav = G_Ofcc[1] + G_H2Ol - G_OOHcav - G_H_e
        #formation of O*(atop) and H2O(l) by OOH*, H+ and e-
        G2aedge = G_Oatop[0] + G_H2Ol - G_OOHedge - G_H_e
        G2acav = G_Oatop[1] + G_H2Ol - G_OOHcav - G_H_e
        #formation of O*(fcc) and OH* by OOH* dissociation
        G2bedge = G_Ofcc[0] + G_OHedge - G_OOHedge
        G2bcav = G_Ofcc[1] + G_OHcav - G_OOHcav
        #formation of OH* by O*(fcc), H+ and e-
        G3edge = G_OHedge - G_Ofcc[0] - G_H_e
        G3cav = G_OHcav - G_Ofcc[1] - G_H_e
        #formation of OH* by O*(atop), H+, and e-
        G3aedge = G_OHedge - G_Oatop[0] - G_H_e
        G3acav = G_OHcav - G_Oatop[1] - G_H_e
        #formation of H2O(l)  by OH*, H+ and e-
        G4edge = G_H2Ol - G_OHedge - G_H_e
        G4cav = G_H2Ol - G_OHcav - G_H_e
        #formation of 2 O*(fcc) from 1 O2(g) by dissociation
        G_O2edge = 2*G_Ofcc[0] - G_O2g
        G_O2cav = 2*G_Ofcc[1] - G_O2g
        #Rate constants and activation energies
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
        kO2edge = kB*T/h*np.exp(-max(G_O2edge+EaO2,EaO2)/(kB*T))
        k_O2edge = kB*T/h*np.exp(-max(-G_O2edge+EaO2,EaO2)/(kB*T))
        kO2cav = kB*T/h*np.exp(-max(G_O2cav+EaO2,EaO2)/(kB*T))
        k_O2cav = kB*T/h*np.exp(-max(-G_O2cav+EaO2,EaO2)/(kB*T))
        #rates
        r1edge=k1edge*(1-OHedge-OOHedge-Ocovatopedge)*pO2*pH2**0.5
        r_1edge = k_1edge*OOHedge
        r1cav=k1cav*(1-OHcav-OOHcav-Ocovatopcav)*pO2*pH2**0.5
        r_1cav = k_1cav*OOHcav
        r2edge = k2edge*OOHedge*pH2**0.5
        r_2edge = k_2edge*Ocovfccedge*pH2O
        r2cav = k2cav*OOHcav*pH2**0.5
        r_2cav = k_2cav*Ocovfcccav*pH2O
        r2aedge = k2aedge*OOHedge*pH2**0.5
        r_2aedge = k_2aedge*Ocovatopedge*pH2O
        r2acav = k2acav*OOHcav*pH2**0.5
        r_2acav = k_2acav*Ocovatopcav*pH2O
        r2bedge = k2bedge*OOHedge
        r_2bedge = k_2bedge*Ocovfccedge*OHedge
        r2bcav = k2bcav*OOHcav
        r_2bcav = k_2bcav*Ocovfcccav*OHcav
        r3edge = k3edge*Ocovfccedge*pH2**0.5
        r_3edge = k_3edge*OHedge
        r3cav = k3cav*Ocovfcccav*pH2**0.5
        r_3cav = k_3cav*OHcav
        r3aedge = k3aedge*Ocovatopedge*pH2**0.5
        r_3aedge = k_3aedge*OHedge
        r3acav = k3acav*Ocovatopcav*pH2**0.5
        r_3acav = k_3acav*OHcav
        r4edge = k4edge*OHedge*pH2**0.5
        r_4edge = k_4edge*(1-OHedge-OOHedge-Ocovatopedge)*pH2O
        r4cav = k4cav*OHcav*pH2**0.5
        r_4cav = k_4cav*(1-OHcav-OOHcav-Ocovatopcav)*pH2O
        rOedge = 2*(kO2edge*pO2*2*(1-Ocovfccedge)**2)
        r_Oedge = 2*(k_O2edge*2*(Ocovfccedge)**2)
        rOcav = 2*(kO2cav*pO2*2*(1-Ocovfcccav)**2)
        r_Ocav = 2*(k_O2cav*2*(Ocovfcccav)**2)
        if GET_RATE == True:
            rate_electron_edge = fsum([r1edge,-r_1edge,r2edge,-r_2edge,r2aedge
                                  ,-r_2aedge,r3edge,-r_3edge,r3aedge,-r_3aedge
                                  ,r4edge,-r_4edge])
            rate_electron_cavity = fsum([r1cav,-r_1cav,r2cav,-r_2cav,r2acav,-r_2acav
            ,r3cav,-r_3cav,r3acav,-r_3acav,r4cav,-r_4cav])
            return rate_electron_cavity,rate_electron_edge
        else:
            #changes in coverage
            dThetaOOHedgedt = fsum([r1edge,-r_1edge,-r2edge,r_2edge,-r2aedge
                                    ,r_2aedge,-r2bedge,r_2bedge])
            dThetaOHedgedt = fsum([r2bedge,-r_2bedge,r3edge,-r_3edge,r3aedge
                                   ,-r_3aedge,-r4edge,r_4edge])
            dThetaOOHcavdt = fsum([r1cav,-r_1cav,-r2cav,r_2cav,-r2acav,r_2acav
                                   ,-r2bcav,r_2bcav])
            dThetaOHcavdt = fsum([r2bcav,-r_2bcav,r3cav,-r_3cav,r3acav,-r_3acav
                                  ,-r4cav,r_4cav])
            dThetaOfccedgedt = fsum([rOedge,-r_Oedge,r2edge,-r_2edge,r2bedge
                                    ,-r_2bedge,-r3edge,r_3edge])
            dThetaOatopedgedt = fsum([r2aedge,-r_2aedge,-r3aedge,r_3aedge])
            dThetaOfcccavdt = fsum([rOcav,-r_Ocav,r2cav,-r_2cav,r2bcav,-r_2bcav
                                    ,-r3cav,r_3cav ])
            dThetaOatopcavdt = fsum([r2acav,-r_2acav,-r3acav,r_3acav])
            dydt = [dThetaOHedgedt,dThetaOHcavdt,dThetaOOHedgedt,dThetaOOHcavdt
                    ,dThetaOfccedgedt,dThetaOatopedgedt,dThetaOfcccavdt,dThetaOatopcavdt]
            return dydt
    
    def gcn_scaling(self,GCN,GCN_reference):
        """
        Get the amount to shift the zero coverage surface energy derivatives due to changes
        in GCN. Uses the energy-GCN relationship fo Calle-Vallejo from his 2015
        Science paper to get the shift in OH and OOH energy. We have found that
        the slope of the O binding energy with GCN of the relevant atop site
        to be 0.08.
        
        Input:
            GCN: scalar
                GCN value to determine the shift in zero coverage
                surface energy derivatives
            GCN_reference: scalar
                GCN of DFT data for which Hamiltonian was parameterized
            
        Output:
            GCN_scaling: array of length 3 
                Amount to shift zero coverage change in sufrace energy [eV] due to difference
                in GCN of desired coverage/rate and the GCN of the DFT data used
                to parameterize the Hamiltonian
        """
        #binding energies at zero coverage for OH and OOH, respecitively, without solvation effects on Pt
        #These binding enregies are taken from the Energy-GCN scaling relation found in
        #Calle Vallejo 2015 (Science)
        x = metal('Pt')
        BEs_zerocov = np.array(x.get_BEs(GCN, uncertainty = False, correlations = False))
        #referecne binding energy of the DFT calculations used in parameterizing the Hamiltonian
        BEs_reference = np.array(x.get_BEs(GCN_reference, uncertainty = False, correlations = False))
        #GCN_scaling is a length two array and contains contains the amount
        #to shift the zero coverage adsorption energy of OH and OOH due to GCN
        #being lower or higher than that at which the Hamiltonian was parameterized
        GCN_scaling =  BEs_zerocov - BEs_reference
        #0.0873 is the scaling of Oxygen adsorption energy in the fcc site
        #as the GCN of the nearest neighbor atop site changes
        GCN_scaling = np.append(GCN_scaling,0.0873*(GCN-GCN_reference))
        return GCN_scaling
        
    def coverage(self,GCN_scaling):
        """
        Solve the coupled nonlinear ODEs to find the steady state coverage
        
        Input:
            GCN_scaling: array of length 3 or shape (2,3)
                Amount to shift zero coverage change in surface energy of
                adsorbates based on GCN    
        Output:
            t: 1d array
                Explicit time steps 
            sol: 1d array
                Coverages at the explicit time steps [ML]
        Attributes used
            site_type: string
                terrace, edge, or cavity_edge
            popt: array of length 9
                Parameters fitted to the Hamiltonian for the terrace/edge 
                site without cavities for determining change in surface energy 
                for oxygen adsorption. Uses DFT data for 6.4 GCn edge sites.
            poptO: array of length 8
                Parameters fitted to the Hamiltonian for the terrace/edge 
                site without cavities for determining repuslive effects of 
                adsorbed oxygen. Uses DFT data afor 6.4 GCN edge sites.
            popt_cavity_edge: array of length 9
                Parameters for O surface energy derivative function based on regressing 
                the coupled edge and cavity site Hamiltonian
        """
        n = range(3,5) #number of external time steps to solve with the ODE integration
        m = range(0,6) #time to end the simulation
        for i in n:
            for ii in m:
                t = np.linspace(0, 10**ii, 10**i)
                if self.site_type == 'cavity_edge':
#==============================================================================
#                     initial guess is based on the steady state coverage for
#                     a 5.1 GCN edge and 8.5 GCN cavity (the coupled cavity/edge
#                     DFT data)
#==============================================================================
                    initial_guess = [  1.90376033e-01,   4.69651644e-04,   4.87155845e-07,
                                     2.51137546e-12,   1.60978814e-01,   8.88361906e-09,
                                     1.13227229e-02,   5.17383971e-12]
#==============================================================================
#                     finds a good initial guess for the coverage by applying 
#                     many time steps at small t so that fewer time steps at 
#                     longer t can be used
#==============================================================================
                    sol = odeint(self.coverage_cavity_edge, initial_guess, np.linspace(0,10**-6,10**6)
                                 , args=(self.popt,self.poptO,self.popt_cavity_edge
                                         ,GCN_scaling[0],GCN_scaling[1]))
                    #rerun simulaiton for longer time with previous solution
                    sol = odeint(self.coverage_cavity_edge, sol[-1], t
                                 , args=(self.popt,self.poptO,self.popt_cavity_edge
                                         ,GCN_scaling[0],GCN_scaling[1]))
                else:
#==============================================================================
#                     initial guess is based on the steady state coverage for
#                     a 7.5 GCN terrace (undefected DFT data)
#==============================================================================
                    initial_guess = [6.14313809e-06, 3.56958665e-12
                                     , 1.93164910e-01, 7.73636912e-12]
#==============================================================================
#                     finds a good initial guess for the coverage by applying 
#                     many time steps at small t so that fewer time steps at 
#                     longer t can be used
#==============================================================================                    
                    sol = odeint(self.coveragefunc, initial_guess, np.linspace(0,10**-6,10**6)
                                 , args=(self.popt,GCN_scaling))
                    #rerun simulaiton for longer time with previous solution
                    sol = odeint(self.coveragefunc, sol[-1], t
                                 , args=(self.popt,GCN_scaling))
#==============================================================================
#                 if the difference between coverages at the last three time 
#                 steps and their respective prior time steps are less then 10**-12
#                 then steady state coverage has been reached.
#==============================================================================
                diffm =  np.abs(sol[-4:-1].ravel() - sol[-3:].ravel())
                if max(diffm) < 10**-12:
                    break
#==============================================================================
#             if there is no jump in coverage greater than 0.5 then the number of
#             time steps provided to the ode solver is sufficient.
#==============================================================================
            diffn = np.abs(sol[1:].ravel()-sol[0:-1].ravel())
            if max(diffn) < 0.5:
                    break
        return t, sol
    
    def rate(self,coverage,GCN_scaling):
        """
        Solve the coupled nonlinear ODEs to find the rate at the provided coverage
        
        Input:
            coverage: array of length 4 or length 8
                Coverage of the terrace/edge system or the edges coupled with
                the cavity
            GCN_scaling: array of length 3 or shape(2,3)
                Amount to shift zero coverage change in surface energy
                of adsorbates based on GCN    rate(self,coverage,GCN_scaling)
        Output:
            rate: 1d array
                Reaction rate (sum of all electrochemical steps) [mA/site]     
        Attributes used
            site_type: string
                Terrace, Edge, or cavity_edge
            popt: array of length 9
                Parameters fitted to the Hamiltonian for the terrace/edge 
                site without cavities for determining change in surface energy 
                for oxygen adsorption. Uses DFT data for 6.4 GCn edge sites.
            poptO: array of length 8
                Parameters fitted to the Hamiltonian for the terrace/edge 
                site without cavities for determining repuslive effects of 
                adsorbed oxygen. Uses DFT data afor 6.4 GCN edge sites.
            popt_cavity_edge: array of length 9
                Parameters for O surface energy derivative function based on regressing 
                the coupled edge and cavity site Hamiltonian
        """
        if self.site_type == 'cavity_edge':
            sol = odeint(self.coverage_cavity_edge, coverage
                    , np.linspace(0, 1, 10**6), args=(self.popt, self.poptO
                    ,self.popt_cavity_edge,GCN_scaling[0],GCN_scaling[1]))
            rate = self.coverage_cavity_edge(sol[-1],'tdummy',self.popt,self.poptO
                    ,self.popt_cavity_edge,GCN_scaling[0],GCN_scaling[1],GET_RATE=True)
            #rerun with smaller time steps if the rate is negative
            if rate <=0:
                sol = odeint(self.coverage_cavity_edge, sol[-1]
                    , np.linspace(0, 0.01, 10**8), args=(self.popt, self.poptO
                    ,self.popt_cavity_edge,GCN_scaling[0],GCN_scaling[1]))
                rate = self.coverage_cavity_edge(sol[-1],'tdummy',self.popt,self.poptO
                    ,self.popt_cavity_edge,GCN_scaling[0],GCN_scaling[1],GET_RATE=True)
            if rate <=0:
                sol = odeint(self.coverage_cavity_edge, sol[-1]
                    , np.linspace(0, 10**-4, 10**8), args=(self.popt, self.poptO
                    ,self.popt_cavity_edge,GCN_scaling[0],GCN_scaling[1]))
                rate = self.coverage_cavity_edge(sol[-1],'tdummy',self.popt,self.poptO
                    ,self.popt_cavity_edge,GCN_scaling[0],GCN_scaling[1],GET_RATE=True)
        else:
            sol = odeint(self.coveragefunc, coverage
                        , np.linspace(0, 1, 10**6), args=(self.popt,GCN_scaling))
            rate = self.coveragefunc(sol[-1],'tdummy',self.popt,GCN_scaling,GET_RATE=True)
            #rerun with smaller time steps if rate is negative
            if rate <=0:
                sol = odeint(self.coveragefunc, sol[-1]
                        , np.linspace(0, 0.01, 10**8), args=(self.popt,GCN_scaling))
                rate = self.coveragefunc(sol[-1],'tdummy',self.popt,GCN_scaling,GET_RATE=True)
            if rate <=0:
                sol = odeint(self.coveragefunc, sol[-1]
                        , np.linspace(0, 10**-4, 10**8), args=(self.popt,GCN_scaling))
                rate = self.coveragefunc(sol[-1],'tdummy',self.popt,GCN_scaling,GET_RATE=True)
        return rate
    
    def get_coverage(self,GCN):
        """
        Interactive method for obtaining an array of coverages where the last
        coverages is the steady state coverage at the provided GCN
        
        Input:
            GCN: scalar
                GCN of the site for which the coverage is desired
        Output:
            t: 1d array
                Explicit time steps 
            sol: 1d array
                Coverages at the explicit time steps [ML]
        Attributes used
            site_type: string
                terrace, edge, or cavity_edge
        """
        if self.site_type == 'terrace':
            GCN_reference = 7.5
            GCN_scaling = self.gcn_scaling(GCN,GCN_reference)
            
            t, sol = self.coverage(GCN_scaling)
        if self.site_type == 'edge':
            GCN_reference = 6.417
            GCN_scaling = self.gcn_scaling(GCN,GCN_reference)
            t, sol = self.coverage(GCN_scaling)
        if self.site_type =='cavity_edge':
            GCN_reference = 8.5
            GCN_scaling_cavity = self.gcn_scaling(GCN[0],GCN_reference)
            GCN_reference = 5.167
            GCN_scaling_edge = self.gcn_scaling(GCN[1],GCN_reference)
            t, sol = self.coverage([GCN_scaling_cavity,GCN_scaling_edge])
        return t, sol
    
    def get_rate(self,GCN,coverage):
        """
        Interactive method for obtaining the rate atspecified GCN and coverage
        
        Input:
            GCN: scalar
                GCN of the site for which the coverage is desired
            Coverage: array of length 4 or length 8
                Coverages for the desired rate
        Output:
            rate: 1d array
                Reaction rate (sum of all electrochemical steps) [mA/site]
        Attributes used
            site_type: string
                terrace, edge, or cavity_edge
        """
        if self.site_type == 'terrace':
            GCN_reference = 7.5
            GCN_scaling = self.gcn_scaling(GCN,GCN_reference)
            rate = self.rate(coverage,GCN_scaling)
        if self.site_type == 'edge':
            GCN_reference = 6.417
            GCN_scaling = self.gcn_scaling(GCN,GCN_reference)
            rate = self.rate(coverage,GCN_scaling)
        if self.site_type == 'cavity_edge':
            GCN_reference = 8.5
            GCN_scaling_cavity = self.gcn_scaling(GCN[0],GCN_reference)
            GCN_reference = 5.167
            GCN_scaling_edge = self.gcn_scaling(GCN[1],GCN_reference)
            rate = self.rate(coverage,[GCN_scaling_cavity,GCN_scaling_edge])
        return rate