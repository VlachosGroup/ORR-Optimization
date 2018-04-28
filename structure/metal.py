"""
Models thermodynamic properties of the metal such as 
generalized coordination number dependent dependent binding energies
"""

import numpy as np
import os

class metal:

    '''
    Class for properties of a metal
    '''
    
    def __init__(self, met_name):
        
        '''
        Pt DFT data from Tables S2 and S3 of F. Calle-Vallejo, J. Tymoczko, 
        V. Colic, Q.H. Vu, M.D. Pohl, K. Morgenstern, D. Loffreda, P. Sautet, 
        W. Schuhmann, and A.S. Bandarenka, Science 350, 185 (2015).
        Au data is taken from Figure Figure S22 of the same paper.
        '''
        
        self.name = met_name
        
        if met_name == 'Pt':
            
            self.E_coh = 4.5222                         # cohesive energy (eV)
            self.lc_PBE = 3.968434601                   # lattice constant for the PBE functional 
            self.load_DFT_data('Pt_BEs.npy')
			
        elif met_name == 'Au':
            
            self.E_coh = 2.3645 				 
            self.lc_PBE = 4.155657928				
            #self.load_DFT_data('Au_BEs.npy')
            self.load_DFT_data('Au_Nature_BEs.npy')
            
        else:
            
            raise ValueError(met_name + ' not found in metal database.')
        
    
    def load_DFT_data(self,np_fname):
        '''
        :param np_fname: Name of the numpy file with binding energy data
        The format is a n x 3 array where n is the number of data points. The first
        column is the GCN of each site. The 2nd and 3rd columns are the OH* and OOH*
        binding energies respectively, referenced to OH(g) and OOH(g)
        '''
        dir = os.path.dirname(__file__)
        np_fname = os.path.join(dir, np_fname)
        BEs = np.load(np_fname)
    
        # Regress OH BE vs. GCN
        self.OH_slope, self.OH_int = np.polyfit(BEs[:,0], BEs[:,1], 1)
        BE_OH_pred = BEs[:,0] * self.OH_slope + self.OH_int
        res_OH = BEs[:,1] - BE_OH_pred          # Compute residuals
        self.sigma_OH_BE = np.std(res_OH)       # Compute variance of residuals
        self.res_OH = res_OH
        
        # Regress OOH BE vs. GCN
        self.OOH_slope, self.OOH_int = np.polyfit(BEs[:,0], BEs[:,2], 1)
        BE_OOH_pred = BEs[:,0] * self.OOH_slope + self.OOH_int
        res_OOH = BEs[:,2] - BE_OOH_pred         # Compute residuals
        self.sigma_OOH_BE = np.std(res_OOH)     # Compute variance of residuals
        self.res_OOH = res_OOH
        
        '''
        Perform PCA on residuals
        '''
        data = np.transpose( np.vstack([res_OH, res_OOH]) )
        eigenvectors, eigenvalues, V = np.linalg.svd(data.T, full_matrices=False)
        projected_data = np.dot(data, eigenvectors)
        sigma = projected_data.std(axis=0)
        self.pca_mat = eigenvectors
        self.pca_inv = np.linalg.inv( self.pca_mat )
        self.sigma_pca_1 = sigma[0]
        self.sigma_pca_2 = sigma[1]
        
    
    def get_BEs(self, GCN, uncertainty = False, correlations = True):
        '''
        :param GCN: generalized binding energy of the site
        :param uncertainty: If true, add random noise due to error in GCN relation
        :param correlations: If true, use PCA as joint PDF of
        :returns: OH and OOH* binding energies
        '''
        
        OH_BE = self.OH_slope * GCN + self.OH_int
        OOH_BE = self.OOH_slope * GCN + self.OOH_int
        
        if uncertainty:
            
            if correlations:
            
                pca1 = self.sigma_pca_1 * np.random.normal()
                pca2 = self.sigma_pca_2 * np.random.normal()
                BE_errors = np.matmul(np.array([pca1, pca2]), self.pca_mat )
                OH_BE_error = BE_errors[0]
                OOH_BE_error = BE_errors[1]
                
            else:
            
                OH_BE_error = self.sigma_OH_BE * np.random.normal()
                OOH_BE_error = self.sigma_OOH_BE * np.random.normal()
            
            OH_BE += OH_BE_error
            OOH_BE += OOH_BE_error
        
        return [OH_BE, OOH_BE]