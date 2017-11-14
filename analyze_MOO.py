'''
Analyze the results of a multi-objective optimization
'''

import os
import numpy as np

dir = '/home/vlachos/mpnunez/ORR_data/optimization/activity_max'

multi_objective_traj = np.load(os.path.join(dir, 'trajectory_a.npy'))
energy_min_traj = np.load(os.path.join(dir, 'trajectory_b.npy'))

print 'After multi-objective optimization'
print 'Current density (mA/cm^2): ' + str(multi_objective_traj[-1,1])
print 'Surface energy (J/m^2): ' + str(multi_objective_traj[-1,2])

print '\nAfter energy minimization'
print 'Current density (mA/cm^2): ' + str(energy_min_traj[-1,1])
print 'Surface energy (J/m^2): ' + str(energy_min_traj[-1,2])