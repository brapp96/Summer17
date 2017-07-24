"""
Comprehensive test file
"""

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

if __name__ == '__main__':
     # setting global variables
    globVars.init()
    globVars.FILEPATH = parent_dir_name+'/results/'
    globVars.DEBUG = True

    # setting parameters
    rw_filename = 'sentences.txt'
    emb_filename = 'emb.txt'
    num_reps = 10
    length = 60
    dim = 50
    winsize = 8
    c_array = [5.0, 10.0, 15.0, 20.0]
    K_array = [2]  # number of communities
    N_array = [100] # number of nodes
    lambda_array = [0.99] # B0 = lambda*I + (1-lambda)*ones(1, 1)
    rand_tests = 2
    algos = ['deep', 'nbt', 'sc', 'abp']
    metrics = ['nmi', 'ccr', 'ars']

    # parsing arguments to file
    usage_str = '''NBtest.py [-q] [-i <infile>] [-o <outfile>] [-w <winsize>]
                   [-d <dimension>] [-r <num_paths>] [-l <length>]'''
    try:
        opts, _ = getopt.getopt(sys.argv[1:], "hqd:w:r:l:i:o:")
    except getopt.GetoptError:
        print usage_str
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print usage_str
            sys.exit(2)
        elif opt == '-q':
            globVars.DEBUG = False
        elif opt == 'd':
            dim = arg
        elif opt == 'w':
            winsize = arg
        elif opt == 'r':
            num_reps = arg
        elif opt == 'l':
            length = arg
        elif opt == 'i':
            rw_filename = arg
        elif opt == 'o':
            emb_filename = arg

    # initialize log file
    if globVars.DEBUG:
        logfile = open(globVars.FILEPATH+'test.log', 'w')
        logfile.write(str(datetime.datetime.utcnow()))
        logfile.write('\nLogging details of run:\n')
        logfile.close()

    # initialize results
    results = {}
    for name in algos:
        results[name] = {}
        for met in metrics:
            results[name][met] = np.empty((len(c_array), len(K_array),
                                           len(N_array), len(lambda_array),
                                           rand_tests))

    # main loop
    for indc, c in enumerate(c_array):
        for indK, K in enumerate(K_array):
            for indN, N in enumerate(N_array):
                for indll, ll in enumerate(lambda_array):
                    globVars.printDebug('\n\nK: '+str(K)+', N: '+str(N)+', c: '
                                        +str(c)+', lambda: '+str(ll))
                    model_sbm1 = SBM.SBM_param_init(K, N, ll, c)
                    for rand in range(rand_tests):
                        y = {}
                        globVars.printDebug('\nBeginning iteration %d of %d...'
                                            % (rand+1, rand_tests))
                        exp_str = 'N'+str(N)+'-K'+str(K)+'-c'+str(c)+'-la'+str(ll)+'-iter'+str(rand)

                        # simulate graph
                        G = SBM.SBM_simulate_fast(model_sbm1)
                        ln, names = SBM.get_label_list(G)

                        # algo1: proposed deepwalk algorithm
                        globVars.printDebug('starting normal VEC algorithm...')
                        model_w2v = algs.SBM_learn_deepwalk(G, rw_filename, emb_filename,
                                                            num_reps=num_reps, dim=dim,
                                                            length=length, winsize=winsize,
                                                            NBT=False)
                        X = model_w2v[names]
                        k_means = KMeans(n_clusters=K, max_iter=100, precompute_distances=False)
                        k_means.fit(X)
                        y['deep'] = k_means.labels_

                        # algo2: nonbacktracking algorithm
                        globVars.printDebug('starting nonbacktracking VEC algorithm...')
                        model_w2v = algs.SBM_learn_deepwalk(G, rw_filename, emb_filename,
                                                            num_reps=num_reps, dim=dim,
                                                            length=length, winsize=winsize,
                                                            NBT=True)
                        X = model_w2v[names]
                        k_means = KMeans(n_clusters=K, max_iter=100, precompute_distances=False)
                        k_means.fit(X)
                        y['nbt'] = k_means.labels_

                        if N <= 2000:
                            # algo3: spectral clustering
                            A = nx.to_scipy_sparse_matrix(G)
                            globVars.printDebug('starting spectral clustering...')
                            sc = SpectralClustering(n_clusters=K, affinity='precomputed',
                                                    eigen_solver='arpack')
                            sc.fit(A)
                            y['sc'] = sc.labels_

                            # algo4: belief propogation
                            globVars.printDebug('starting ABP algorithm...')
                            r = 3
                            m, mp, lambda1 = ABP.abp_params(model_sbm1)
                            y['abp'] = ABP.SBM_ABP(G, r, lambda1, m, mp)

                        # save results
                        for name in algos:
                            if N > 2000 and (name == 'sc' or name == 'abp'): continue
                            m = {}
                            m['nmi'], m['ccr'], m['ars'] = algs.cal_metrics(ln, y[name])
                            for met in metrics:
                                results[name][met][indc, indK, indN, indll, rand] = m[met]
    
    # Write results to file
    params = {'n': N_array, 'k': K_array, 'c': c_array, 'l': lambda_array,
            'iter': rand_tests, 'algorithms': algos, 'metrics': metrics}
    savename = "%spkls/fulltest.pkl" % globVars.FILEPATH
    savename2 = "%spkls/fulltestparams.pkl" % globVars.FILEPATH
    pickle.dump(results, open(savename, 'wb'), protocol=2)
    pickle.dump(params, open(savename2, 'wb'), protocol=2)
    

