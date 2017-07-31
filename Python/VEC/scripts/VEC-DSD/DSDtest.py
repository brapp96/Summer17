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
import time 

len_rw = 15     # Length of random walks
quiet = True   # True if less output needed
N =  100
K = 2
c_array = [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0]
#c_array = [4.0,5.0, 6.0, 8.0, 10.0,12.0, 15.0, 20.0]
ll = 0.9
rand = 0 

start = time.time()
acc_ccr = np.zeros(len(c_array))
acc_nmi = np.zeros(len(c_array)
        )
for i in range(len(c_array)):
    c = c_array[i]
    exp_str = 'N'+str(N)+'-K'+str(K)+'-c'+str(c)+'-la'+str(ll)+'-iter'+str(rand)
    
    with open('../../graphs/'+exp_str+'.txt') as fp:
        Gstr = fp.readlines()
    G = nx.read_adjlist(Gstr[:-1])
    labels = Gstr[-1][1:-1].split(',')
    true_labels = np.array([int(ch) for ch in labels])
    
    order = (np.arange(N)).astype(str)
    graphAdj = nx.to_numpy_matrix(G,order, dtype=int) 
    graphAdj = np.asarray(graphAdj)
    
    
    
    ## Load graph from file
    #print('Loading graph from file...')
    #graphAdj = np.loadtxt('graph')
    #true_labels = np.loadtxt('graph_labels')
    #N = np.size(graphAdj[0])
    #pdb.set_trace()
    
    #Calculate DSD and obtain similarity matrix for spectral clustering
    print('Calculate DSD...')
    DSD = calcDSD.calculator(graphAdj, true_labels, len_rw, quiet) 
  #  pdb.set_trace()
    DSD[DSD == 0] = 1
    DSD_sim = 1/(DSD)
    
    # Apply spectral clustering and reorder labels using Hungarian algorithm
    print('Applying spectral clustering...')
    labels = sc.spectral_clustering(DSD_sim, n_clusters=2)
    Conf = metrics.confusion_matrix(true_labels, labels)
    #pdb.set_trace()
    row, col = scipy.optimize.linear_sum_assignment(-1*Conf)
    #print(r)
    #print(c)
    
    
    
    # Get metrics
    print('Calculating metrics...')
    acc_nmi[i] = metrics.normalized_mutual_info_score(true_labels,labels)
    acc_ccr[i] = float(Conf[row, col].sum())/float(N)
    
print('\nSBM parameters: N={}, k={}, c={}, lambda={}\n\n'.format(N, K, c_array, ll))
    
print('CCR: {}\n'.format(acc_ccr))
print('NMI: {}\n\n'.format(acc_nmi))
    #pdb.set_trace()

f = open('results', 'a')
f.write(str(acc_ccr)+ '\n')
f.write(str(acc_nmi))
f.close()

print('Time elapsed: {}\n'.format(time.time() - start))

