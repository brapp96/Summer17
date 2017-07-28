# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 14:15:12 2017

@author: brian
"""
import os
import sys
import itertools
import networkx as nx
parent_dir_name = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if parent_dir_name not in sys.path:
    sys.path.append(parent_dir_name)
from src import SBMlib as SBM

c_array = [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0]
K_array = [2]  # number of communities
N_array = [100, 200, 500, 1000, 2000, 5000, 10000] # number of nodes
lambda_array = [0.9] # B0 = lambda*I + (1-lambda)*ones(1, 1)
rand_tests = 5
    
for [c,K,N,ll] in itertools.product(c_array,K_array,N_array,lambda_array):
    model_sbm = SBM.SBM_param_init(K, N, ll, c)
    for rand in range(rand_tests):
        G = SBM.SBM_simulate_fast(model_sbm)
        ln, names = SBM.get_label_list(G)
        fname = '../graphs/N'+str(N)+'-K'+str(K)+'-c'+str(c)+'-la'+str(ll)+'-iter'+str(rand)+'.txt'
        nx.write_adjlist(G, fname)
        fp = open(fname, 'ab')
        fp.write(str(ln))
        fp.close()