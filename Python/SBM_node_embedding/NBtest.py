"""
Create Figure, compare three algorithms across

collecting results from nmi, ccr, ars, and modularity
"""

import pickle
import itertools
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from src import SBMlib as SBM
from src import VEClib as algs
from src import ABPlib as ABP

# MACRO parameter setting
rw_filename = 'sentences.txt'
emb_filename = 'emb.txt'
num_paths = 10
length_path = 60
emb_dim = 50
winsize = 8
quiet = True

if __name__ == '__main__':
    # generating multiple graphs for the same parameter setting
    rand_tests = 5
    # setting storage space for results
    nmi_arry = {}
    ccr_arry = {}
    ars_arry = {}
    # parameter setting
    c_array = [3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0]
    K_array = [2]  # number of communities
    N_array = [1000, 2000, 5000, 10000] # number of nodes
    lambda_array = [0.9] # B0 = lambda*I + (1-lambda)*ones(1,1)
    # scanning through parameters
    for N in N_array:
        for c, K, lambda_n in itertools.product(c_array, K_array, lambda_array):
            if not quiet: print 'K:', K, 'N:', N, 'c:', c, 'lambda:', lambda_n
            model_sbm1 = SBM.SBM_param_init(K, N, lambda_n, c)
            for rand in range(rand_tests):
                if not quiet: print 'Beginning iteration', rand+1, 'of', rand_tests, '...'
                strsub1 = 'K'+str(K)+'N'+str(N)+'c'+str(c)+'la'+str(lambda_n)+'rd'+str(rand) # for saving results
                # simulate graph
                G = SBM.SBM_simulate_fast(model_sbm1)
                ln, nodeslist = SBM.get_label_list(G)

                # algo1: proposed deepwalk algorithm
                if not quiet: print 'starting normal VEC algorithm...'
                model_w2v = algs.SBM_learn_deepwalk(G, rw_filename, emb_filename)
                X = model_w2v[nodeslist]
                k_means = KMeans(n_clusters=K, max_iter=100, precompute_distances=False)
                k_means.fit(X)
                y_our = k_means.labels_
                nmi_arry, ccr_arry, ars_arry = algs.summary_res(nmi_arry, ccr_arry, ars_arry, ln, y_our, 'deep', 'c', c, rand)

                # algo2: nonbacktracking algorithm
                if not quiet: print 'starting NBRW VEC algorithm...'
                model_w2v = algs.SBM_learn_deepwalk(G, rw_filename, emb_filename, NBT=True)
                X = model_w2v[nodeslist]
                k_means = KMeans(n_clusters=K, max_iter=100, precompute_distances=False)
                k_means.fit(X)
                y_nbt = k_means.labels_
                nmi_arry, ccr_arry, ars_arry = algs.summary_res(nmi_arry, ccr_arry, ars_arry, ln, y_nbt, 'nbt', 'c', c, rand)

                # algo3: spectral clustering
                A = nx.to_scipy_sparse_matrix(G)
                if not quiet: print 'starting spectral clustering...'
                sc = SpectralClustering(n_clusters=K, affinity='precomputed', eigen_solver='arpack')
                sc.fit(A)
                y_sc = sc.labels_
                nmi_arry, ccr_arry, ars_arry = algs.summary_res(nmi_arry, ccr_arry, ars_arry, ln, y_sc, 'sc', 'c', c, rand)

                # algo4: belief propogation
                if not quiet: print 'starting ABP algorithm...'
                r = 3
                m, mp, lambda1 = ABP.abp_params(model_sbm1)
                y_abp = ABP.SBM_ABP(G, r, lambda1, m, mp)
                nmi_arry, ccr_arry, ars_arry = algs.summary_res(nmi_arry, ccr_arry, ars_arry, ln, y_abp, 'abp', 'c', c, rand)
        savename = "exp1%d.pkl" % N
        res = [nmi_arry, ccr_arry, ars_arry]
        pickle.dump(res, open(savename, 'wb'), protocol=2)

