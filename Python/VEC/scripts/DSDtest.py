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
import calcDSD
import scipy
from sklearn import metrics
import networkx as nx
import matplotlib.pyplot as plt

len_rw = 15     # Length of random walks
quiet = True   # True if less output needed

# Load graph from file
print('Loading graph from file...')
graphAdj = np.loadtxt('graph')
true_labels = np.loadtxt('graph_labels')
N = np.size(graphAdj[0])
#pdb.set_trace()

#Calculate DSD and obtain similarity matrix for spectral clustering
print('Calculate DSD...')
DSD = calcDSD.calculator(graphAdj, true_labels, len_rw, quiet) 
DSD_sim = 1/(DSD + np.eye(N))

# Apply spectral clustering and reorder labels using Hungarian algorithm
print('Applying spectral clustering...')
labels = sc.spectral_clustering(DSD_sim, n_clusters=2)
Conf = metrics.confusion_matrix(true_labels, labels)
#pdb.set_trace()
r, c = scipy.optimize.linear_sum_assignment(-1*Conf)
#print(r)
#print(c)
#pdb.set_trace()


# Get metrics
print('Calculating metrics...')
acc_nmi = metrics.normalized_mutual_info_score(true_labels,labels)
acc_ccr = float(Conf[r, c].sum())/float(N)

print('CCR: {}\n'.format(acc_ccr))
print('NMI: {}\n\n'.format(acc_nmi))
#pdb.set_trace()
    
