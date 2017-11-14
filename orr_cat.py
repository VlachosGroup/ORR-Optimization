'''
Computes catalyst properties specific to ORR catalysts
'''

import numpy as np
import copy
import random

from ase.neighborlist import NeighborList

from metal import metal
from ORR import ORR_rate
from graph_theory import Graph
from dynamic_cat import dynamic_cat

class orr_cat(dynamic_cat):
    
    '''
    Oxygen reduction reaction catalyst structure with defects
    '''
    
    def __init__(self, met_name = 'Pt', facet = '111', dim1 = 12, dim2 = 12):
        
        dynamic_cat.__init__(self, met_name = met_name, facet = facet, dim1 = dim1, dim2 = dim1, fixed_layers = 3, variable_layers = 1)       # Call parent class constructor
        
        self.metal = None
        
        self.template_graph = None
        self.defected_graph = None
        self.active_atoms = None            # Atoms which contribute to the current density
        
        if facet == '111':
            self.active_CN = 9                      # CN must be less than or equal to this to be active    
        elif facet == '100':
            self.active_CN = 8
            
        self.active_atoms = range(2 * self.atoms_per_layer, 4 * self.atoms_per_layer)
        self.metal = metal(met_name)
        
        # Compute normalization factor from volcano plot
        GCN_vec = np.linspace(3,10)
        volcano = np.zeros(GCN_vec.shape)
        for ind in xrange(len(GCN_vec)):
            BEs = self.metal.get_BEs(GCN_vec[ind])
            BE_OH = BEs[0]
            BE_OOH = BEs[1]
            volcano[ind] = ORR_rate(BE_OH, BE_OOH)
        
        self.i_max = np.max(volcano)
        
        
        '''
        Build template graph
        '''
        
        # Find neighbors based on distances
        rad_list = ( 2.77 + 0.2 ) / 2 * np.ones(len(self.atoms_template))               # list of neighboradii for each site
        neighb_list = NeighborList(rad_list, self_interaction = False)      # set bothways = True to include both ways
        neighb_list.build(self.atoms_template)
        
        self.template_graph = Graph()
        for i in range(len(self.atoms_template)):
            self.template_graph.add_vertex(i)
        
        for i in range(len(self.atoms_template)):
            for j in neighb_list.neighbors[i]:
                self.template_graph.add_edge([i,j])
                
        self.defected_graph = copy.deepcopy(self.template_graph)
        
        self.occs_to_atoms()
        self.occs_to_graph()
    
    
    def graph_to_occs(self):
        '''
        Convert graph representation of defected structure to occupancies
        '''
        self.variable_occs = [0 for i in range(len(self.variable_atoms))]
        ind = 0
        for i in self.variable_atoms:
            if self.defected_graph.is_node(i):
                self.variable_occs[ind] = 1
            ind += 1
            
            
    def occs_to_graph(self, x = None):
        '''
        Build graph from occupancies
        '''
        if x is None:
            x = self.variable_occs
        else:
            self.variable_occs = x
        
        self.defected_graph = self.template_graph.copy_data()
        for i in range(len(x)):
            if x[i] == 0:
                self.defected_graph.remove_vertex(self.variable_atoms[i])
        
    
    def get_Nnn(self):
        '''
        For each active atom, print the number of nearest neighbors that are also active
        '''
        defected_graph = self.defected_graph
        for i in self.active_atoms:
            if defected_graph.is_node(i):
                if defected_graph.get_coordination_number(i) <= self.active_CN:
                    
                    gcn = defected_graph.get_generalized_coordination_number(i, 12)
                    
                    Nnn = 0
                    for j in defected_graph.get_neighbors(i):
                        if j in self.active_atoms:
                            if defected_graph.is_node(j):
                                if defected_graph.get_coordination_number(j) <= self.active_CN:
                                    Nnn += 1
                    
                    print [gcn, Nnn]
        
    
    def get_site_data(self):
        '''
        Evaluate the contribution to the current from each site
        
        :returns: Array site currents for each active site
        '''

        curr_list = [0. for i in range(len(self.active_atoms))]
        for i in range(len(self.active_atoms)):
            site_ind = self.active_atoms[i]
            if self.defected_graph.is_node(site_ind):
                if self.defected_graph.get_coordination_number(site_ind) <= self.active_CN:
                    gcn = self.defected_graph.get_generalized_coordination_number(site_ind, 12)
                    BEs = self.metal.get_BEs(gcn)
                    BE_OH = BEs[0]
                    BE_OOH = BEs[1]
                    curr_list[i] = ORR_rate(BE_OH, BE_OOH)
                    
        curr_list = np.transpose( np.array(curr_list).reshape([2,self.atoms_per_layer]) )  
        return curr_list
        
                    
    def eval_current_density(self, normalize = True):
        
        '''
        :param normalize: current density [mA/cm^2]
        :returns: Total current (mA) or Current density [mA/cm^2]
        '''
        
        site_currents = self.get_site_data()
        I = np.sum(site_currents)
        
        if normalize:
            return self.normalize_current_density(I)
        else:
            return I
    
        
    def normalize_current_density(self,I):
        '''
        :param I: total current in mA
        :returns: Total current (mA) or Current density [mA/cm^2]
        '''
        square_cm_per_square_angstrom = 1.0e-16         # conversion factor
        return I / ( self.surface_area * square_cm_per_square_angstrom)
        
    
    def eval_surface_energy(self, normalize = True):
        
        '''
        Normalized: surface energy [J/m^2]
        Not normalized: formation energy [eV] or surface energy (J/m^2)
        '''
        
        E_form = 0
        for i in self.active_atoms:
            if self.defected_graph.is_node(i):
                E_form += self.metal.E_coh * ( 1 - np.sqrt( self.defected_graph.get_coordination_number(i) / 12.0 ) )
                
        if normalize:
            return self.normalize_surface_energy(E_form)
        else:
            return E_form      
    

    def normalize_surface_energy(self,E_form):
        '''
        :param E_form: formation energy of the slab (eV)
        :returns: Current density [mA/cm^2]
        '''
        ev_to_Joule = 1.60218e-19                       # conversion factor
        square_m_per_square_angstrom = 1.0e-20          # conversion factor
        return E_form * ev_to_Joule / ( self.surface_area * square_m_per_square_angstrom )
        
    
    def flip_atom(self, ind):
        
        '''
        If atom number ind is present in the defected graph, remove it.
        If it is not present, add it and all edges to adjacent atoms.
        '''        
        super(orr_cat, self).flip_atom(ind)     # Call super class method to change the occupancy vector
        
        if self.defected_graph.is_node(ind):
            self.defected_graph.remove_vertex(ind)
        else:
            self.defected_graph.add_vertex(ind)
            for neighb in self.template_graph.get_neighbors(ind):
                if self.defected_graph.is_node(neighb):
                    self.defected_graph.add_edge([ind, neighb])
                    
                    
    def rand_move_CE(self, move_these = None):
        ''' Randomly change an adjacent atom-occupancy pair '''
        
        # Identify an adjacent atom-occupancy pair
        if move_these is None:
        
            # Enumerate atom-occupancies adjacent pairs
            pair_list = []
            for i in self.variable_atoms:
                if self.defected_graph.is_node(i):
                    vacant_neighbs = []
                    for j in self.template_graph.get_neighbors(i):
                        if not self.defected_graph.is_node(j):
                            pair_list.append([i,j])
            
            if not pair_list == []:
                move_these = random.choice(pair_list)    
        
        # Flip these occupancies
        if not move_these is None:
            self.flip_atom(move_these[0])
            self.flip_atom(move_these[1])
        
        return move_these