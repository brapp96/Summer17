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
    read_graphs = True    
    
    c_array = [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0,15.0,20.0]
    K_array = [2]  # number of communities
    N_array = [100] # number of nodes
    lambda_array = [0.9] # B0 = lambda*I + (1-lambda)*ones(1, 1)
    rand_tests = 1
    algos = ['deep', 'nbt']
    metrics = ['nmi', 'ccr']

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
        print str(datetime.datetime.utcnow())
        print str('Logging details of run:')
    logfile = open(globVars.FILEPATH+'test.log', 'w')
    logfile.write(str(datetime.datetime.utcnow()))
    logfile.write('\nLogging details of run:')
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
                    model_sbm = SBM.SBM_param_init(K, N, ll, c)
                    for rand in range(rand_tests):
                        y = {}
                        globVars.printDebug('\nBeginning iteration %d of %d...'
                                            % (rand+1, rand_tests))
                        exp_str = 'N'+str(N)+'-K'+str(K)+'-c'+str(c)+'-la'+str(ll)+'-iter'+str(rand)
                        
                        if read_graphs:
                            with open('../graphs/'+exp_str+'.txt') as fp:
                                Gstr = fp.readlines()
                            G = nx.read_adjlist(Gstr[:-1])
                            labels = Gstr[-1][1:-1].split(',')
                            ln = [int(ch) for ch in labels]
                            names = [str(i) for i in range(N)]
                        else:
                            # simulate graph
                            G = SBM.SBM_simulate_fast(model_sbm)
                            ln, names = SBM.get_label_list(G)
                            
                            # write graph to file 
                            m = nx.to_numpy_matrix(G, dtype=int) 
                            np.savetxt('graph', m, fmt='%d')
                        
                            lbls = np.array(ln)
                            np.savetxt('graph_labels', lbls, fmt='%d')
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
                            m, mp, lambda1 = ABP.abp_params(model_sbm)
                            y['abp'] = ABP.SBM_ABP(G, r, lambda1, m, mp)

                        # save results
                        for name in algos:
                            if N > 2000 and (name == 'sc' or name == 'abp'): continue
                            m = {}
                            m['nmi'], m['ccr'], m['ars'] = algs.cal_metrics(ln, y[name])
                            for met in metrics:
                                results[name][met][indc, indK, indN, indll, rand] = m[met]


# Print out results
print('\nSBM parameters: N={}, k={}, c={}, lambda={}\n\n'.format(N_array,
    K_array, c_array, lambda_array))
print('Backtracking RW:\n')    
print('CCR : {}'.format(results['deep']['ccr'][:,0,0,0,0]))
print('NMI : {}\n'.format(results['deep']['nmi'][:,0,0,0,0]))

print('Non-backtracking RW:\n')
print('CCR : {}'.format(results['nbt']['ccr'][:,0,0,0,0]))
print('NMI : {}\n\n'.format(results['nbt']['nmi'][:,0,0,0,0]))


# print metrics to file
f = open('VEC-DSD/results', 'a')
import pdb
pdb.set_trace()
f.write(str(results['deep']['ccr'][:,0,0,0,0])+ '\n')
f.write(str(results['deep']['nmi'][:,0,0,0,0])+ '\n')
f.write(str(results['nbt']['ccr'][:,0,0,0,0])+ '\n')
f.write(str(results['nbt']['nmi'][:,0,0,0,0])+ '\n')

f.close()



#    # Write results to file
#    params = {'n': N_array, 'k': K_array, 'c': c_array, 'l': lambda_array,
#            'iter': rand_tests, 'algorithms': algos, 'metrics': metrics}
#    savename = "%spkls/fulltest.pkl" % globVars.FILEPATH
#    savename2 = "%spkls/fulltestparams.pkl" % globVars.FILEPATH
#    pickle.dump(results, open(savename, 'wb'), protocol=2)
#    pickle.dump(params, open(savename2, 'wb'), protocol=2)
    

