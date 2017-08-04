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

len_rw = [5, -1]     # Length of random walks
quiet = True   # True if less output needed
N =  10000
K = 2
#c_array = [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0]
c_array = [5.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0]
ll = 0.9
rand_tests = 1

data_const = 'N{}'.format(N)
data_varied = 'C'

f = open('results/dsd_data/{}vs{}.txt'.format(data_const, data_varied), 'a')

for r in range(rand_tests): 
    for dsd_type in len_rw:
        acc_ccr = np.zeros(len(c_array))
        acc_nmi = np.zeros(len(c_array))
                
        for i in range(len(c_array)):
            c = c_array[i]
            exp_str = 'N'+str(N)+'-K'+str(K)+'-c'+str(c)
            
            with open('../VEC/graphs/'+exp_str+'.txt') as fp:
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
            
            #Calculate DSD and obtain similarity matrix for spectral clustering
            print('Calculate DSD...')
            DSD, true_labels = calcDSD.calculator(graphAdj, true_labels, dsd_type, quiet) 
            irow, icol = np.where(DSD == 0)
            DSD[irow, icol] = 1
            DSD_sim = 1/(DSD + np.eye(DSD.shape[0]))
            
            # Apply spectral clustering and reorder labels using Hungarian algorithm
            print('Applying spectral clustering...')

            labels = sc.spectral_clustering(DSD_sim, n_clusters=2)
            Conf = metrics.confusion_matrix(true_labels, labels)
            row, col = scipy.optimize.linear_sum_assignment(-1*Conf)
            
            # Get metrics
            print('Calculating metrics...')
            acc_nmi[i] = metrics.normalized_mutual_info_score(true_labels,labels)
            acc_ccr[i] = float(Conf[row, col].sum())/float(N)
            
           
        #print('\nSBM parameters: N={}, k={}, c={}, lambda={}\n\n'.format(N, K, c_array, ll))
        #    
        #print('CCR: {}\n'.format(acc_ccr))
        #print('NMI: {}\n\n'.format(acc_nmi))
            
        f.write(str(acc_ccr.tolist())[1:-1]+ '\n')
        f.write(str(acc_nmi.tolist())[1:-1]+ '\n')
    

f.write('\n\n\nSBM parameters: N={}, k={}, c={}, lambda={}\n'.format(N, K,c_array, ll))
f.write('DSD parameters: Length of RW={}, RandTests={}'.format(len_rw, rand_tests)) 
f.close()




