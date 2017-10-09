import numpy as np
import pdb

N = 5885
DSD = np.loadtxt('results_converged.DSD1', skiprows=1, usecols=range(1,N));
DSD_sim = 1/(DSD + np.eye(DSD.shape[0]))
pdb.set_trace()
np.savetxt('DSD_sim.txt','DSD_sim')


#pdb.set_trace()

