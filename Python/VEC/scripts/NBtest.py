"""
Basic test file
"""

import os, sys, datetime
import getopt
import pickle
import itertools
import networkx as nx
from sklearn.cluster import KMeans, SpectralClustering
parent_dir_name = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if parent_dir_name not in sys.path:
    sys.path.append(parent_dir_name)
from src import globVars, SBMlib as SBM, VEClib as algs, ABPlib as ABP

def main(argv):
    "Main function"
    # MACRO parameter setting
    rw_filename = 'sentences.txt'
    emb_filename = 'emb.txt'
    num_reps = 10
    length = 60
    emb_dim = 50
    winsize = 8

    # setting global variables
    globVars.init()
    globVars.FILEPATH = parent_dir_name+'/results/'
    globVars.DEBUG = True
    usage_str = '''NBtest.py [-q] [-i <infile>] [-o <outfile>] [-w <winsize>]
                    [-d <dimension>] [-r <num_paths>] [-l <length>]'''
    try:
        opts, _ = getopt.getopt(argv, "hqd:w:r:l:")
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
            emb_dim = arg
        elif opt == 'w':
            winsize = arg
        elif opt == 'r':
            num_reps = arg
        elif opt == 'l':
            length = arg

    if globVars.DEBUG:       
        logfile = open(globVars.FILEPATH+'test.log', 'w')
        logfile.write(str(datetime.datetime.utcnow()))
        logfile.write('\nLogging details of run:\n')
        logfile.close()

    # generating multiple graphs for the same parameter setting
    rand_tests = 3
    # setting storage space for results
    nmi = {}
    ccr = {}
    ars = {}
    # parameter setting
    c_array = [3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0]
    K_array = [2, 3]  # number of communities
    N_array = [100, 200] # number of nodes
    lambda_array = [0.9] # B0 = lambda*I + (1-lambda)*ones(1, 1)
    # scanning through parameters
    for c, K, N, lambda_n in itertools.product(c_array, K_array,
                                               N_array, lambda_array):
        globVars.printDebug('\n\nK: '+str(K)+', N: '+str(N)+', c: '+str(c)\
                            +', lambda: '+str(lambda_n))
        model_sbm1 = SBM.SBM_param_init(K, N, lambda_n, c)
        for rand in range(rand_tests):
            globVars.printDebug('\nBeginning iteration %d of %d...' % 
                                (rand+1, rand_tests))
            exp_str = 'N'+str(N)+';K'+str(K)+';c'+str(c)+';la'+str(lambda_n)\
                      +';iter'+str(rand)
            # simulate graph
            G = SBM.SBM_simulate_fast(model_sbm1)
            ln, nodeslist = SBM.get_label_list(G)

            # algo1: proposed deepwalk algorithm
            globVars.printDebug('starting normal VEC algorithm...')
            model_w2v = algs.SBM_learn_deepwalk(G, rw_filename, emb_filename,
                                                num_reps=num_reps, dim=emb_dim,
                                                length=length, winsize=winsize)
            X = model_w2v[nodeslist]
            k_means = KMeans(n_clusters=K, max_iter=100, precompute_distances=False)
            k_means.fit(X)
            y_deep = k_means.labels_

            # algo2: nonbacktracking algorithm
            globVars.printDebug('starting NBRW VEC algorithm...')
            model_w2v = algs.SBM_learn_deepwalk(G, rw_filename, emb_filename,
                                                num_reps=num_reps, dim=emb_dim,
                                                length=length, winsize=winsize,
                                                NBT=True)
            X = model_w2v[nodeslist]
            k_means = KMeans(n_clusters=K, max_iter=100,
                             precompute_distances=False)
            k_means.fit(X)
            y_nbt = k_means.labels_

            # algo3: spectral clustering
            A = nx.to_scipy_sparse_matrix(G)
            globVars.printDebug('starting spectral clustering...')
            sc = SpectralClustering(n_clusters=K, affinity='precomputed',
                                    eigen_solver='arpack')
            sc.fit(A)
            y_sc = sc.labels_

            # algo4: belief propogation
            globVars.printDebug('starting ABP algorithm...')
            r = 3
            m, mp, lambda1 = ABP.abp_params(model_sbm1)
            y_abp = ABP.SBM_ABP(G, r, lambda1, m, mp)
            
            # save results
            for name, res in [['deep', y_deep], ['nbt', y_nbt], ['sc', y_sc], ['abp', y_abp]]:
                nmi, ccr, ars = algs.summary_res(nmi, ccr, ars, ln, res, name, exp_str)    
        savename = "%s%s%s.pkl" % (globVars.FILEPATH, 'pkls/', exp_str)
        res = [nmi, ccr, ars]
        pickle.dump(res, open(savename, 'wb'), protocol=2)

if __name__ == '__main__':
    main(sys.argv[1:])
