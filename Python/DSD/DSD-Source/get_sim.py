import numpy as np

N = 5885
DSD = np.loadtxt('results_converged.DSD1', skiprows=1, usecols=range(1,N));
np.savetxt('dsd.txt',DSD,fmt='%f');
