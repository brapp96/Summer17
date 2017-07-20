"""

ABP Library

Functions for the ABP algorithm in Abbe et al., 2015, 2016
- SBM_ABP(G, r, lambda1, m, mp)
abp for two communities
- multi_abp(G, r, lambda1, m, mp, dim, K)
abp for mutiple communities
- abp_params(md)
get algorithm parameters for ABP

"""

import networkx as nx
from sklearn.cluster import KMeans
import numpy as np
from . import globVars

def SBM_ABP(G, r, lambda1, m, mp):
    "Runs the ABP algorithm for comparison with VEC."
    # step 1: initialize Y(v, v')(0)
    t = 1
    elist = G.edges()
    cstr = 'cycles<='+str(r)
    Y = {}
    for e in elist:
        v = e[0]
        vp = e[1]
        erev = (vp, v)
        Y[e] = {}
        Y[e][t] = np.random.normal(0, 1)  # y(v, v')(1)
        Y[erev] = {}
        Y[erev][t] = np.random.normal(0, 1) # y(v', v)(1)
    # step 2: check if v, v' is part of a cycle <=r
    for e in elist:
        v = e[0]
        vp = e[1]
        erev = (vp, v)
        iscycle, z = check_cycle(G, v, vp, r)
        Y[e][cstr] = {}
        Y[e][cstr]['bin'] = iscycle
        if iscycle:
            Y[e][cstr]['path'] = z
        Y[erev][cstr] = {}
        Y[erev][cstr]['bin'] = iscycle
        if iscycle:
            Y[erev][cstr]['path'] = z
    # step 3: iterations to calculate y(v, v')(t), 1<t<=m
    elist = Y.keys()
    for t in range(2, m+1):
        for e in elist:
            v = e[0]
            vp = e[1]
            wts = [Y[(vp, vpp)][t-1] for vpp in G[vp] if vpp not in [v]]
            if Y[e][cstr]['bin'] is False:
                # v,v' not in a cycle of length <=r, use eq a) to get y(v,v')(t)
                Y[e][t] = sum(wts)
            else:
                # v v' in a cycle of length r' (where r'<=r)
                z = Y[e][cstr]['path']
                rp = len(z)
                # get the adj to v
                if z[0] == v:
                    vppp = z[1]
                else:
                    vppp = z[-2]
                # if cycle r'==t, use eq c) to get y(v,v')(t)
                if rp == t:
                    Y[e][t] = sum(wts) - float(len(wts))*Y[(vppp, v)][1]
                else:
                    mu = t- rp
                    if mu < 1:
                        Y[e][t] = sum(wts)
                    else:
                        wts2 = [Y[(v, vpp)][mu] for vpp in G[v] if vpp not in [vp]
                                and vpp not in [vppp]]
                        Y[e][t] = sum(wts) - sum(wts2)
    # step 4: get the Y matrix
    Ymat = {}
    nds = G.nodes()
    for v in nds:
        Ymat[v] = {}
        for t in range(1, m+1):
            yts = [Y[(v, s)][t] for s in G[v]]
            Ymat[v][t] = sum(yts)
    # step 5: calculate y', and clustering
    M = np.diag([-1*lambda1]*(m-1), k=1)+np.eye(m)
    em = np.zeros((m, 1))
    em[m-1] = 1
    for _ in range(1, mp):
        em = np.dot(M, em)
    labels_est = []
    for v in nds:
        tmp = [Ymat[v][t+1]*em[t][0] for t in range(0, m)]
        Ymat[v]['yp'] = sum(tmp)
        if Ymat[v]['yp'] > 0.0:
            labels_est.append(1)
        else:
            labels_est.append(0)
    return labels_est

def check_cycle(G, u, v, r):
    "Checks if G contains a cycle including u and v of length less than r."
    # u,v is path of cycle <=r <=> u,v has shortest path<=r-1 after removing (u,v)
    ispartcycle = False
    if r != 1:
        # save a copy of current edge
        edgetmp = {}
        entry = G[u][v]
        for key in entry:
            edgetmp[key] = entry[key]
        # delete current edge from graph
        G.remove_edge(u, v)
        # check if shortest path <= r-1
        if nx.has_path(G, u, v):
            z = nx.shortest_path(G, u, v)
            ispartcycle = len(z) <= r
        # add edge back to graph
        G.add_edge(u, v)
        for key in edgetmp:
            G[u][v][key] = edgetmp[key]
    if ispartcycle:
        return ispartcycle, z
    else:
        return ispartcycle, []

def abp_params(md):
    """
    Creates the parameters for the ABP algorithm.
    Note that this only works for simulated data.
    """
    snr, lambda1, lambda2 = SBM_SNR(md)
    n = md['N']
    m = 2.0*np.log(float(n))/np.log(snr)
    m = int(np.ceil(m)) + 1
    if m < 0: m = 2
    mp = m*np.log(lambda1*lambda1/lambda2/lambda2)/np.log(float(n))
    mp = int(np.ceil(mp)) + 1
    if mp < 0: mp = 2
    return m, mp, lambda1

def multi_abp(G, r, lambda1, m, mp, dim, K):
    "Performs ABP on multiple length paths from 1 to dim"
    N = len(G.nodes())
    mt = np.zeros((N, dim))
    for k in range(dim):
        print 'k-th iter:', k
        y_abp = SBM_ABP(G, r, lambda1, m, mp)
        mt[:, k] = y_abp
    k_means = KMeans(n_clusters=K, max_iter=100, precompute_distances=False)
    k_means.fit(mt)
    y = k_means.labels_
    return y

def SBM_SNR(model):
    "Defines the SNR and first and second eigenvalues of the model"
    Q = model['B0']*model['alpha']*float(model['N'])
    P = np.diag(model['a'])
    Z = np.dot(P, Q)
    u, _ = np.linalg.eig(Z)
    ua = sorted(u, reverse=True)
    globVars.printDebug('lambda1: '+str(ua[0])+'; lambda2: '+str(ua[1]))
    SNR = ua[1]*ua[1]/ua[0]
    return SNR, ua[0], ua[1]
