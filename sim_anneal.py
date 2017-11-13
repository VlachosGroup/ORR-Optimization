'''
Multi-objective simulated annealing optimization
See June 22 "Implemented optimization" commit
'''

import numpy as np
import random
import time

def optimize(cat, weight = 1.0, ensemble = 'GCE', n_cycles = 100, c = 0.7, n_record = 100, verbose = True):
    
    '''
    Use simulated annealing to optimize defects on the surface
    
    :param cat: Catalyst structure
    :param weight: Current density maximization versus surface energy minimization
    :param cat: Object to be optimized. Must have get_OF(), rand_move(), and revert_last() methods.
    :param total_steps: Number of Metropolis steps to run
    :param initial_T: Initial temperature (dimensionless). A linear cooling schedule is used
    :param n_record: number of steps to print out (not counting initial state)
    '''
    
    total_steps = n_cycles * cat.atoms_per_layer
    
    # Evaluate initial structure
    j = cat.eval_current_density()
    SE = cat.eval_surface_energy()
    OF = -weight * j + (1-weight) * SE
    
    steps_to_record = np.linspace(0, total_steps,  n_record+1 )
    step_rec = steps_to_record.astype(int)
    OF_rec_1 = np.zeros(n_record+1)
    OF_rec_2 = np.zeros(n_record+1)
    record_ind = 0
    
    # Record data
    if verbose:
        print 'Steps elapsed \t Current density (J/cm^2) \t Surface energy (mA/cm^2)'
        print str(0) + '\t' + str(j) + '\t' + str(SE)
    OF_rec_1[record_ind] = j
    OF_rec_2[record_ind] = SE
    record_ind += 1
    
    CPU_start = time.time()        
    
    for step in range( total_steps ):
                    
        Metro_temp = c / np.log(step+2)             # Set temperature
        j_prev = j
        SE_prev = SE
        OF_prev = OF                                                        # Record data before changing structure
        if ensemble == 'GCE':
            cat.rand_move()                                                # Do a Metropolis move
        elif ensemble == 'CE':
            atoms_moved = cat.rand_move_CE()
        else:
            raise NameError('Unrecognized ensemble')
        
        # Evaluate the new structure and determine whether or not to accept
        j = cat.eval_current_density()
        SE = cat.eval_surface_energy()
        OF = -weight * j + (1-weight) * SE                  
        
        if OF - OF_prev < 0:                # Downhill move
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
            
            j = j_prev
            SE = SE_prev
            OF = OF_prev            # Use previous values for evaluations
        
        # Record data
        if (step+1) in step_rec:
            if verbose:
                print str(step+1) + '\t' + str(j) + '\t' + str(SE)
            OF_rec_1[record_ind] = j
            OF_rec_2[record_ind] = SE
            record_ind += 1
            
            
    CPU_end = time.time()
    print('Time elapsed: ' + str(CPU_end - CPU_start) )
    
    return np.transpose( np.vstack([step_rec, OF_rec_1, OF_rec_2]) )