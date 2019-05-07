'''
Computes catalyst properties specific to ORR catalysts
'''

import numpy as np
import copy
import random
import os

from ase.neighborlist import PrimitiveNeighborList

from orr_optimizer.metal import metal
from orr_optimizer.ORR import ORR_rate
from orr_optimizer.orr_mkm import *
from orr_optimizer.graph_theory import Graph
from orr_optimizer.dynamic_cat import dynamic_cat
import math

class orr_cat(dynamic_cat):

    '''
    Oxygen reduction reaction catalyst structure with defects
    '''

    def __init__(self, met_name = 'Pt', facet = '111', dim1 = 12, dim2 = 12, volcano = 'JL'):

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
        self.volcano_data = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'all_volcanos.npy'))        # maximum is edge sites at unrealistically high GCN

        self.volcano = volcano
        if volcano == 'JL':
            self.i_max = np.max(self.volcano_data[:,3])     # Others
            #self.i_max = np.max(self.volcano_data[:,1::])
        elif volcano == 'CV':
            self.i_max = np.max(self.volcano_data[:,5])     # Calle-Vallejo only
        else:
            raise NameError('Unrecognized volcano')

        '''
        Build template graph
        '''

        # Find neighbors based on distances
        rad_list = ( 2.77 + 0.2 ) / 2 * np.ones(len(self.atoms_template))               # list of neighboradii for each site
        neighb_list = PrimitiveNeighborList(rad_list, self_interaction = False)      # set bothways = True to include both ways
        neighb_list.build([True,True,True], self.atoms_template.get_cell(), self.atoms_template.get_positions())

        self.template_graph = Graph()
        for i in range(len(self.atoms_template)):
            self.template_graph.add_vertex(i)

        for i in range(len(self.atoms_template)):
            for j in neighb_list.neighbors[i]:
                self.template_graph.add_edge([i,j])

        self.defected_graph = copy.deepcopy(self.template_graph)

        self.occs_to_atoms()
        self.occs_to_graph()

        # Get adjacent indices of cavities near top layer edges
        self.edge_cavity_dict = {}
        apl = self.atoms_per_layer
        for ind in range(apl):
            d1, d2 = self.var_ind_to_sym_inds(ind)
            cav_ind_1 = self.sym_inds_to_var_ind(d1-1, d2+1)
            cav_ind_2 = self.sym_inds_to_var_ind(d1+1, d2+1)
            cav_ind_3 = self.sym_inds_to_var_ind(d1+1, d2-1)
            self.edge_cavity_dict[ind+3*apl] = [cav_ind_1+2*apl, cav_ind_2+2*apl, cav_ind_3+2*apl]


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
        :param x: site occupancies
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

                    gcn = defected_graph.get_generalized_coordination_number(i)

                    Nnn = 0
                    for j in defected_graph.get_neighbors(i):
                        if j in self.active_atoms:
                            if defected_graph.is_node(j):
                                if defected_graph.get_coordination_number(j) <= self.active_CN:
                                    Nnn += 1

                    print ([gcn, Nnn])


    def get_site_currents(self, hist_info = False):
        '''
        Evaluate the contribution to the current from each site
        :param hist_info: If false, just returns the site rates. If true, returns the GCN as well

        :returns: Array site currents for each active site
        '''

        curr_list = [0. for i in self.active_atoms]
        GCN_list = [None for i in self.active_atoms]
        site_categ_list = [None for i in self.active_atoms]
        for i in range(len(self.active_atoms)):
            site_ind = self.active_atoms[i]
            if self.defected_graph.is_node(site_ind):
                if self.defected_graph.get_coordination_number(site_ind) <= self.active_CN:
                    gcn = self.defected_graph.get_generalized_coordination_number(site_ind)

                    if self.volcano == 'JL':
                        if i < self.atoms_per_layer:                                      # bottom layer
                            if gcn > 7.9:                                                   # cavity, not sure of the appropritate cutoff
                                site_type_rates = self.volcano_data[:,4]
                                site_categ_list[site_ind-2*self.atoms_per_layer] = 'cavity'
                            else:                                                           # terrace
                                site_type_rates = self.volcano_data[:,1]
                                site_categ_list[site_ind-2*self.atoms_per_layer] = 'bot terrace'
                        else:                                                               # top layer
                            if gcn > 6.0:  # terrace
                                site_type_rates = self.volcano_data[:,1]
                                site_categ_list[site_ind-2*self.atoms_per_layer] = 'top terrace'
                            else:                                                           # edge
                                there_is_a_nearby_cavity = False
                                possible_cavity_sites = self.edge_cavity_dict[site_ind]
                                for cav_site in possible_cavity_sites:
                                    if site_categ_list[cav_site-2*self.atoms_per_layer] == 'cavity':
                                        there_is_a_nearby_cavity = True

                                if there_is_a_nearby_cavity:
                                    site_type_rates = self.volcano_data[:,3]    # cavity_edge
                                    site_categ_list[site_ind-2*self.atoms_per_layer] = 'cavity edge'
                                else:
                                    site_type_rates = self.volcano_data[:,2]    # edge with no cavity, terrace-like
                                    site_categ_list[site_ind-2*self.atoms_per_layer] = 'edge'

                    elif self.volcano == 'CV':
                        site_type_rates = self.volcano_data[:,5]
                    else:
                        raise NameError('Volcano not set')

                    # interpolate data to get the rate
                    GCN_list[i] = gcn
                    curr_list[i] = np.exp( np.interp( gcn, self.volcano_data[:,0], np.log(site_type_rates) ) )
                    if math.isnan(curr_list[i]):
                        curr_list[i] = 0

        if hist_info:
            GCN_list_new = []
            curr_list_new = []
            for i in range(len(self.active_atoms)):
                if not GCN_list[i] is None:
                    GCN_list_new.append(GCN_list[i])
                    curr_list_new.append(curr_list[i])
            return [GCN_list_new, curr_list_new]
        else:
            curr_list = np.transpose( np.array(curr_list).reshape([2,self.atoms_per_layer]) )
            return curr_list


    def eval_current_density(self, normalize = True):

        '''
        :param normalize: current density [mA/cm^2]
        :returns: Total current (mA) or Current density [mA/cm^2]
        '''

        site_currents = self.get_site_currents()
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
        Evaluate the surface energy of the slab
        :param normalize: Normalized: surface energy [J/m^2]. Not normalized: formation energy [eV] or surface energy (J/m^2)
        :returns: The surface energy, in units depending on whether it is normalized
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
        :param ind: index of the atom to be flipped
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
        '''
        Randomly change an adjacent atom-occupancy pair
        :param move_these: Can specify the atoms to be flipped
        :returns: A two-element list with the indices of the atoms in the pair
        '''

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
