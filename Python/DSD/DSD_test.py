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


num_rw = 50
quiet = False

graphAdj = np.loadtxt('graph')
true_labels = np.loadtxt('graph_labels', dtype=int)
N = np.size(graphAdj[0])

# remedy isolated nodes
for i in range(N):
    if ~np.any(graphAdj[i, :]):
        cluster = true_labels[i]
        neighbors = np.where(true_labels == cluster)
        pdb.set_trace()


names = {}
for i in xrange(1, N+1):
    names[('Pro%04d' % i)] = i-1
names = collections.OrderedDict(sorted(names.items(),
                                           key=lambda x: x[1]))

DSD = calcDSD.calculator(graphAdj, num_rw, quiet)

# compute Gaussian kernel     
sigma = 1 
data = np.exp(-DSD**2 / (2.*(sigma**2)))

# apply spectral clustering and reorder labels
spectral = sc.SpectralClustering(n_clusters = 2, affinity='precomputed')
spectral.fit(data)
labels = spectral.fit_predict(data) 
pdb.set_trace()
