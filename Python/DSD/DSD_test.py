# Main DSD clustering test file
#
# Uses DSD to generate a new graph from input adjacency matrix and 
# uses spectral clustering to identify clusters.
#
# 7/25/2017, Anuththari Gamage
#

import numpy as np
import sklearn.cluster as sc
import pdb
import collections
import calcDSD

num_rw = 5      # Number of random walks
quiet = False   # True is less output needed

# Load graph from file
graphAdj = np.loadtxt('smallPPIgraph.txt')
true_labels = np.loadtxt('smallPPIlabels.txt', dtype=str)
N = np.size(graphAdj[0])
    
#Calculate DSD and reset isolated node distances
DSD = calcDSD.calculator(graphAdj, num_rw, quiet)
inf_marker = np.max(DSD)
DSD[np.where(DSD == -1)] = inf_marker

# Compute Gaussian kernel as similarity graph for spectral clustering  
sigma = 1 
DSD_sim = np.exp(-DSD**2 / (2.*(sigma**2)))

# Apply spectral clustering and reorder labels \\\TO DO
labels = sc.spectral_clustering(DSD_sim)
pdb.set_trace()
