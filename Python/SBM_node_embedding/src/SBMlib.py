"""
This file contains all the functions for simulating SBM models

## setting SBM model parameters
- alias_setup(probs): sets up the aliasing parameters
- alias_draw(J,q): chooses an element from a discrete distribution
- SBM_param_init(K, N, lambda_n, alpha_n, dataType='const'): 1/n scaling scheme default

## simulate random graph with a SBM model
- SBM_simulate_fast(model): use only this one since it is efficient for large sparse graphs

## other util functions
- SBM_savemat(G, edgefilename, nodefilename): save SBM graphs in edge and node file
- SBM_SNR(model): calculate the SNR defined in Abbe et al., 2016

"""

import math
import networkx as nx
import numpy as np
import numpy.random as npr

def alias_setup(probs):
    """
    This function is to help draw random samples from discrete distribution with specific weights,
    the code were adapted from the following source:
    https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/

    arguments:
    probs: the discrete probability
    return:
    J, q: temporary lists to assist drawing random samples
    """
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int)
    # Sort the data into the outcomes with probabilities
    # that are larger and smaller than 1/K.
    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K*prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)
    # Loop though and create little binary mixtures that
    # appropriately allocate the larger outcomes over the
    # overall uniform mixture.
    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] - (1.0 - q[small])

        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)
    return J, q

def alias_draw(J, q):
    """
    This function is to help draw random samples from discrete distribution with specific weights,
    the code were adapted from the following source:
    https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/

    arguments:
    J, q: generated from alias_setup(prob)
    return:
    a random number ranging from 0 to len(prob)
    """
    # Draw from the overall uniform mixture.
    kk = int(npr.rand()*len(J))
    # Draw from the binary mixture, either keeping the
    # small one, or choosing the associated larger one.
    if npr.rand() < q[kk]:
        return kk
    else:
        return J[kk]

#%%
def SBM_param_init(K, N, lambda_n, alpha_n, dataType='const', **prob_weight):
    """
    create SBM model parameters
    dataType determines the type of scaling done on the SBM
    community weights default to be balanced (p = [1/k,...1/k]) but can be
    given as an optional argument (which doesn't need to be normalized but
    does need to have k indices)
    """
    SBM_params = {}
    SBM_params['K'] = K
    SBM_params['N'] = N
    SBM_params['lambda_n'] = float(lambda_n)
    if dataType == 'const':
        SBM_params['alpha'] = alpha_n/float(N)
    elif dataType == 'log':
        SBM_params['alpha'] = alpha_n*math.log(N)/float(N)
    else:
        raise NameError('dataType must be "const" or "log"')
    SBM_params['B0'] = lambda_n * np.eye(K) + (1-lambda_n)*np.ones((K, K))
    if 'p' in prob_weight:
        z = prob_weight['p']
        SBM_params['a'] = z/sum(z)
    else:
        z = np.ones((1, K))
        SBM_params['a'] = z[0]/float(K)
    return SBM_params

#%%
def SBM_simulate(model, quiet=True):
    """
    simulate SBM graph
    model is returned by SBM_param_init() function
    """
    G = nx.Graph()
    b = model['a']
    J, q = alias_setup(b)
    n = model['N']
    B = model['B0']*model['alpha']
    totaledges = 0
    # add nodes with communities attributes
    for key in range(n):
        comm = alias_draw(J, q)
        G.add_node(key, community=comm)
    # sample edges
    for i in range(n):
        com1 = G.node[i]['community']
        for j in range(i+1, n):
            com2 = G.node[j]['community']
            prob = B[com1, com2]
            s = np.random.binomial(1, prob, 1)
            if s[0] == 1:
                G.add_edge(i, j, weight=1.0)
                totaledges += 1
    if not quiet: print 'the graph has', totaledges, 'edges in total'
    return G

def SBM_simulate_fast(model, quiet=True):
    G = nx.Graph()
    b = model['a']
    J, q = alias_setup(b)
    n = model['N']
    k = model['K']
    B = model['B0']*model['alpha']
    totaledges = 0
    # add nodes with communities attributes
    grps = {}
    for t in range(k):
        grps[t] = []
    for key in range(n):
        comm = alias_draw(J, q)
        G.add_node(key, community=comm)
        grps[comm].append(key)
    for i in range(k):
        grp1 = grps[i]
        L1 = len(grp1)
        for j in range(i, k):
            grp2 = grps[j]
            L2 = len(grp2)
            if i == j:
                Gsub = nx.fast_gnp_random_graph(L1, B[i, i])
            else:
                Gsub = nx.algorithms.bipartite.random_graph(L1, L2, B[i, j])
            for z in Gsub.edges():
                nd1 = grp1[z[0]]
                nd2 = grp2[z[1]-L1]
                G.add_edge(nd1, nd2, weight=1.0)
                totaledges += 1
    if not quiet: print 'the graph has', totaledges, 'edges in total'
    return G

def SBM_savemat(G, edgefilename, nodefilename):
    """
    writing down G graph by its nodes attributes and
    """
    nx.write_edgelist(G, edgefilename, data=False)
    nodeslist = G.nodes()
    with open(nodefilename, 'w') as fwrite:
        for key in nodeslist:
            fwrite.write(str(key)+' '+str(G.node[key]['community'])+'\n')
    return 1

def SBM_SNR(model, quiet=True):
    """
    help to define the SNR and lambda1
    """
    Q = model['B0']*model['alpha']*float(model['N'])
    P = np.diag(model['a'])
    Z = np.dot(P, Q)
    u, _ = np.linalg.eig(Z)
    ua = sorted(u, reverse=True)
    if not quiet: print 'lambda1:', ua[0], '\nlambda2:', ua[1]
    SNR = ua[1]*ua[1]/ua[0]
    return SNR, ua[0], ua[1]


