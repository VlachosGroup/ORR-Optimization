'''
Multi-objective simulated annealing optimization
'''

import numpy as np
import random
import time

def optimize(cat, weight = 1.0, ensemble = 'GCE', n_cycles = 100, T_0 = 0.7,
    j_norm = 1.0, se_norm = 1.0, n_record = 100, verbose = True):
    
    '''
    Use simulated annealing to optimize defects on the surface
    
    :param cat: Catalyst structure
    :param ensemble: GCE - number of top layer atoms is not constant, CE - number of top layer atoms is constant
    :param weight: Current density maximization versus surface energy minimization
    :param n_cycles: Multiply by number of top layer atoms to get the total number of Metropolis steps
    :param T_0: Numerical parameter for temerature schedule
    :param j_norm: Normalization factor for current density
    :param se_norm: Normalization factor for surface energy
    :param n_record: Number of time points to record during optimization
    :param verbose: Print data at each time point
    '''
    
    total_steps = n_cycles * cat.atoms_per_layer
    tau = total_steps / 5.0
    
    # Evaluate initial structure
    total_current = cat.eval_current_density(normalize = False)
    E_form = cat.eval_surface_energy(normalize = False)
    OF = -weight * total_current / j_norm + (1-weight) * E_form / se_norm
    
    steps_to_record = np.linspace(0, total_steps,  n_record+1 )
    step_rec = steps_to_record.astype(int)
    OF_rec_1 = np.zeros(n_record+1)
    OF_rec_2 = np.zeros(n_record+1)
    record_ind = 0
    
    # Record data
    current_density = cat.normalize_current_density(total_current)
    surface_energy = cat.normalize_surface_energy(E_form)
    if verbose:
        print ('Steps elapsed \t Current density (mA/cm^2) \t Surface energy (J/m^2)')
        print (str(0) + '\t' + str(current_density) + '\t' + str(surface_energy))
    OF_rec_1[record_ind] = current_density
    OF_rec_2[record_ind] = surface_energy
    record_ind += 1
    
    CPU_start = time.time()        
    
    for step in range( total_steps ):
                    
        Metro_temp = T_0  * np.exp( - step / tau )             # Set temperature
        j_prev = total_current
        E_form_prev = E_form
        OF_prev = OF                                                        # Record data before changing structure
        if ensemble == 'GCE':
            cat.rand_move()                                                # Do a Metropolis move
        elif ensemble == 'CE':
            atoms_moved = cat.rand_move_CE()
        else:
            raise NameError('Unrecognized ensemble')
        
        # Evaluate the new structure and determine whether or not to accept
        total_current = cat.eval_current_density(normalize = False)
        E_form = cat.eval_surface_energy(normalize = False)
        OF = -weight * total_current / j_norm + (1-weight) * E_form / se_norm                  
        
        if OF - OF_prev < 0 or Metro_temp == np.inf:                # Downhill move or infinite temperature
            accept = True
        else:                               # Uphill move
            if Metro_temp > 0:              # Finite temperature, may accept uphill moves
                accept = np.exp( - ( OF - OF_prev ) / Metro_temp ) > random.random()
            else:                           # Zero temperature, never accept uphill moves
                accept = False
        
        # Reverse the change if the move is not accepted
        if not accept:
        
            if ensemble == 'GCE':
                cat.revert_last()       # Do a Metropolis move
            elif ensemble == 'CE':
                cat.rand_move_CE(move_these = atoms_moved)
            else:
                raise NameError('Unrecognized ensemble')
            
            total_current = j_prev
            E_form = E_form_prev
            OF = OF_prev            # Use previous values for evaluations
        
        # Record data
        if (step+1) in step_rec:
            current_density = cat.normalize_current_density(total_current)
            surface_energy = cat.normalize_surface_energy(E_form)
            if verbose:
                print (str(step+1) + '\t' + str(current_density) + '\t' + str(surface_energy))
            OF_rec_1[record_ind] = current_density
            OF_rec_2[record_ind] = surface_energy
            record_ind += 1
            
            
    CPU_end = time.time()
    print('Time elapsed: ' + str(CPU_end - CPU_start) )
    
    return np.transpose( np.vstack([step_rec, OF_rec_1, OF_rec_2]) )