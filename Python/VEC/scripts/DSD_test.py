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
import scipy
from sklearn import metrics

num_rw = 5      # Number of random walks
quiet = False   # True is less output needed

# Load graph from file
print('Loading graph from file...')
graphAdj = np.loadtxt('graph.txt')
true_labels = np.loadtxt('graph_lbls.txt')
N = np.size(graphAdj[0])
    
#Calculate DSD and reset isolated node distances
print('Calculate DSD...')
DSD = calcDSD.calculator(graphAdj, num_rw, quiet)
inf_marker = np.max(DSD)
#pdb.set_trace()
DSD[np.where(DSD == -1)] = inf_marker

# Compute Gaussian kernel as similarity graph for spectral clustering  
sigma = 1 
DSD_sim = np.exp(-DSD**2 / (2.*(sigma**2)))
#pdb.set_trace()

# Apply spectral clustering and reorder labels \\\TO DO
print('Applying spectral clustering...')
labels = sc.spectral_clustering(DSD_sim, n_clusters=2)
#pdb.set_trace()

# Get metrics
print('Calculating metrics...')
acc_nmi = metrics.normalized_mutual_info_score(true_labels,labels)
Conf = metrics.confusion_matrix(true_labels, labels)
r, c = scipy.optimize.linear_sum_assignment(-1*Conf)
acc_ccr = float(Conf[r, c].sum())/float(N)

print('CCR')
print(acc_ccr)
print('NMI')
print(acc_nmi)


