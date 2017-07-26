# Main DSD clustering test file
#
# Uses DSD to generate a new graph from input adjacency matrix and 
# uses spectral clustering to identify clusters.
#
# 7/25/2017, Anuththari Gamage
#

import numpy as np
import sklearn.cluster as sc

# load DSD matrix from file
data = np.loadtxt('graph_DSD.DSD1', skiprows=1,
        usecols=(np.arange(1,941)));
vertices = np.loadtxt('graph_DSD.DSD1', skiprows=1, usecols=(0), dtype=str)
sorted_idx = np.argsort(vertices) # obtain correct indices  since vertices 
                                  # are out of order in the DSD matrix
   
# compute Gaussian kernel     
sigma = 5 
data = np.exp(-data**2 / (2.*(sigma**2)))


# apply spectral clustering and reorder labels
spectral = sc.SpectralClustering(n_clusters = 2, affinity='precomputed')
spectral.fit(data)
labels = spectral.fit_predict(data) 
true_labels = np.zeros(len(labels), dtype=int)  # contains labels in order of
                                                # vertex (1,...,n)
for i in range(len(labels)):
    true_labels[i] = labels[sorted_idx[i]]

np.savetxt('dsd_labels', true_labels, fmt='%d')


