'''
Plots the probability of a random site of a given structure being
more active than a random site with optimal GCN
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mat

from savitzky_golay import *

metal_name = 'Au'
data = np.load(metal_name + '_compare_data.npy')
x_vec = data[0,:]
y_vec = data[1,:]

mat.rcParams['mathtext.default'] = 'regular'
mat.rcParams['text.latex.unicode'] = 'False'
mat.rcParams['legend.numpoints'] = 1
mat.rcParams['lines.linewidth'] = 2
mat.rcParams['lines.markersize'] = 12

plt.figure()
plt.plot(x_vec, y_vec)
plt.ylim(0,0.5)
plt.xticks(size=24)
plt.yticks(size=24)
plt.xlabel('Generalized coordination number',size=24)
plt.ylabel('Probability of\nbeating optimum',size=24)
plt.tight_layout()
plt.savefig(metal_name + '_p_test.png')
plt.close()