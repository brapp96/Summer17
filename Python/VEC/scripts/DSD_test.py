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

len_rw = -1     # Length of random walks
quiet = False   # True is less output needed

# Load graph from file
print('Loading graph from file...')
graphAdj = np.loadtxt('graph')
true_labels = np.loadtxt('graph_labels')
N = np.size(graphAdj[0])

#G = nx.Graph(graphAdj)
#positions = nx.random_layout(G, dim=2)
#nx.draw_networkx_edges(G, pos=positions, edge_color='#000000',width=1)        
#plt.show()

#Calculate DSD and reset isolated node distances
print('Calculate DSD...')
DSD = calcDSD.calculator(graphAdj, true_labels, len_rw, quiet)





pdb.set_trace()
#inf_marker = np.max(DSD)
#DSD[np.where(DSD == -1)] = inf_marker

# Compute Gaussian kernel as similarity graph for spectral clustering  
sigma = 100 
DSD_sim = np.exp(-DSD**2 / (2.*(sigma**2)))
#pdb.set_trace()

# Apply spectral clustering and reorder labels \\\TO DO
print('Applying spectral clustering...')
labels = sc.spectral_clustering(DSD, n_clusters=2)
pdb.set_trace()

# Get metrics
print('Calculating metrics...')
acc_nmi = metrics.normalized_mutual_info_score(true_labels,labels)
Conf = metrics.confusion_matrix(true_labels, labels)
r, c = scipy.optimize.linear_sum_assignment(-1*Conf)
acc_ccr = float(Conf[r, c].sum())/float(N)

print('CCR: {}\n'.format(acc_ccr))
print('NMI: {}\n\n'.format(acc_nmi))
#pdb.set_trace()

