## Test file for PPI networks

import os
import sys
import datetime
import getopt
import cPickle as pickle
import networkx as nx
from sklearn.cluster import KMeans, SpectralClustering
import numpy as np
parent_dir_name = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if parent_dir_name not in sys.path:
    sys.path.append(parent_dir_name)
from src import globVars, SBMlib as SBM, VEClib as algs, ABPlib as ABP
import pdb

if __name__ == '__main__':
     # setting global variables
    globVars.init()
    globVars.FILEPATH = parent_dir_name+'/results/'
    globVars.DEBUG = True

# setting parameters
rw_filename = 'PPIrw.txt'
emb_filename = 'PPIemb.txt'
num_reps = 10
length = 60
dim = 50
winsize = 8
rand_tests = 2
algos = ['deep', 'nbt']
metrics = ['nmi', 'ccr']
K = 8;


npG = np.loadtxt('graph')
G = nx.Graph(npG)           # Graph used
names = np.arange(1000)
names = names.astype(str)   # Vertex labels
ln = np.loadtxt('graph_labels')    # true labels
y = {}
#pdb.set_trace()

# algo1: proposed deepwalk algorithm
print('starting normal VEC algorithm...')
model_w2v = algs.SBM_learn_deepwalk(G, rw_filename, emb_filename, num_reps=num_reps, dim=dim, length=length, winsize=winsize, NBT=False)
X = model_w2v[names]
#pdb.set_trace()
k_means = KMeans(n_clusters=K, max_iter=100, precompute_distances=False)
k_means.fit(X)
y['deep'] = k_means.labels_

# algo2: nonbacktracking algorithm
print('starting nonbacktracking VEC algorithm...')
model_w2v = algs.SBM_learn_deepwalk(G, rw_filename, emb_filename,
                                    num_reps=num_reps, dim=dim,
                                    length=length, winsize=winsize,
                                    NBT=True)
X = model_w2v[names]
k_means = KMeans(n_clusters=K, max_iter=100, precompute_distances=False)
k_means.fit(X)
y['nbt'] = k_means.labels_

# see results

for name in algos:
    print(name)
    m = {}
    m['nmi'], m['ccr'], m['ars'] = algs.cal_metrics(ln, y[name])
    for met in metrics:
       print(met)
       print(m[met])


