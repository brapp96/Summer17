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

ppbAdj = np.loadtxt('graph')
M = int(sum(sum(ppbAdj))/2)
N = np.size(ppbAdj[0])

names = {}
for i in xrange(1, N+1):
    names[('Pro%04d' % i)] = i-1
names = collections.OrderedDict(sorted(names.items(),
                                           key=lambda x: x[1]))

DSD = calcDSD.calculator(ppbAdj,50 , False)

# compute Gaussian kernel     
sigma = 1 
data = np.exp(-DSD**2 / (2.*(sigma**2)))

# apply spectral clustering and reorder labels
spectral = sc.SpectralClustering(n_clusters = 2, affinity='precomputed')
spectral.fit(data)
labels = spectral.fit_predict(data) 
gtlabels = np.loadtxt('graph_labels')
pdb.set_trace()
