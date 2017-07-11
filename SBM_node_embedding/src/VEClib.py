"""
This file contains all the functions for node embedding

## from graph to embedding:
- SBM_learn_deepwalk_1(G, num_paths, length_path, emb_dim, rw_filename, emb_filename, winsize)
write paths in to file and then optimize
- SBM_learn_deepwalk_2(G, num_paths, length_path, emb_dim, rw_filename, emb_filename, winsize)
write paths in to file and then optimize using c-code implementation
- SBM_learn_deepwalk_3(G, num_paths, length_path, emb_dim, rw_filename, emb_filename, winsize)
save paths in memory and then optimize


## from path to embedding:
- SBM_learn_fromcorpus_1(rw_filename, emb_dim, winsize,  emb_filename):
read already existed paths and optimize
- SBM_learn_fromcorpus_2(rw_filename, emb_dim, winsize,  emb_filename):
read already existed paths and optimize



"""
#import math
import numpy as np
import networkx as nx
import gensim.models.word2vec as w2v
from sklearn import metrics
import scipy
import SBMlib as SBM

import NBlib

#############################
def build_node_alias(G):
    """
    build dictionary S that is easier to generate random walks on G
    G is networkx objective
    return: nodes_rw with J, q for each node created using alias_draw functions
    """
    nodes = G.nodes()
    nodes_rw = {}
    for nd in nodes:
        d = G[nd]
        entry = {}
        entry['names'] = [key for key in d]
 #       weights = [math.exp(d[key]['weight']) for key in d]
        weights = [d[key]['weight'] for key in d]
        sumw = sum(weights)
        entry['weights'] = [i/sumw for i in weights]
        J,q = SBM.alias_setup(entry['weights'])
        entry['J'] = J
        entry['q'] = q
        nodes_rw[nd] = entry
    return nodes_rw

def create_rand_walks(S, num_paths, length_path, filename):
    """
    S = from the build_node_alias
    filename - where to write results
    using exp(rating) as edge weight
    """
    fwrite = open(filename,'w')
    nodes = S.keys()
    for nd in nodes:
        for i in range(num_paths):
            walk = [nd] # start as nd
            for j in range(length_path):
                cur = walk[-1]
                next_nds = S[cur]['names']
                if len(next_nds)<1:
                    break
                else:
                    J = S[cur]['J']
                    q = S[cur]['q']
                    rd = SBM.alias_draw(J,q)
                    nextnd = next_nds[rd]
                    walk.append(nextnd)
            walk = [str(x) for x in walk]
            fwrite.write(" ".join(walk) + '\n')
    fwrite.close()
    return 1

def create_rand_walks_inmem(S, num_paths, length_path):
    """
    S = from the build_node_alias
    filename - where to write results
    using exp(rating) as edge weight
    """
    sentence = []
    nodes = S.keys()
    for nd in nodes:
        for i in range(num_paths):
            walk = [nd] # start as nd
            for j in range(length_path):
                cur = walk[-1]
                next_nds = S[cur]['names']
                if len(next_nds)<1:
                    break
                else:
                    J = S[cur]['J']
                    q = S[cur]['q']
                    rd = SBM.alias_draw(J,q)
                    nextnd = next_nds[rd]
                    walk.append(nextnd)
            walk = [str(x) for x in walk]
            sentence.append(walk)
    return sentence

#%%

def SBM_learn_writecorpus1(G, num_paths, length_path, rw_filename):
    print '1 building alias auxiliary functions'
    S = build_node_alias(G)
    print '2 creating random walks'
    create_rand_walks(S, num_paths, length_path, rw_filename)
    return 1

def SBM_learn_fromcorpus_1(rw_filename, emb_dim, winsize,  emb_filename):
    print '3 learning word2vec models'
    sentence = w2v.LineSentence(rw_filename)
    model_w2v = w2v.Word2Vec(sentence, size=emb_dim, window=winsize, \
                             min_count=0, sg = 1, negative = 5, sample = 1e-1, workers=4, iter = 3)
    return model_w2v

def SBM_learn_fromcorpus_2(rw_filename, emb_dim, winsize,  emb_filename):
    import os
    print '3 learning word2vec models using C code'
    comman = './word2vec -train '+rw_filename+' -output '+emb_filename+' -size '+str(emb_dim)+' -window '+str(winsize)+' -negative 5 -cbow 0 -min-count 0 -iter 5 -sample 1e-1'
    os.system(comman)
    model_w2v = w2v.Word2Vec.load_word2vec_format(emb_filename, binary=False)
    return model_w2v


def SBM_learn_deepwalk_1(G, num_paths, length_path, emb_dim, rw_filename, emb_filename, winsize):
    """
    learning SBM model through deepwalk, using gensim package
    File I/O involved:
    first write all the randwalks on disc, then read to learn word2vec
    speed is relatively slow, but scales well to very large dataset
    Inputs:
    G: graph
    num_paths: number of random walks starting from each node
    length_path: length of each random walk
    rw_filename: file name to store the created corpus of sentences from graph G
    emb_filename: file name to store the learned embeddings of the nodes
    emb_dim: the dimensionality of the embeddings
    """
    print '1 building alias auxiliary functions'
    S = build_node_alias(G)
    print '2 creating random walks'
    NBlib.create_rand_walks_NB(S, num_paths, length_path, rw_filename)
    print '3 learning word2vec models'
    sentence = w2v.LineSentence(rw_filename)
    model_w2v = w2v.Word2Vec(sentence, size=emb_dim, window=winsize, min_count=0, sg = 1, negative = 5, sample = 1e-1, workers=5, iter=3)
#    print '4 saving learned embeddings'
#    model_w2v.save_word2vec_format(emb_filename)
    return model_w2v

def SBM_learn_deepwalk_2(G, num_paths, length_path, emb_dim, rw_filename, emb_filename, winsize):
    """
    learning SBM model through deepwalk
    Using the word2vec C implementation, some computational tricks involved.
    Writing random walks on file
    can scale to large dataset well
    Inputs:
    G: graph
    num_paths: number of random walks starting from each node
    length_path: length of each random walk
    rw_filename: file name to store the created corpus of sentences from graph G
    emb_filename: file name to store the learned embeddings of the nodes
    emb_dim: the dimensionality of the embeddings
    """
    import os
    print '1 building alias auxiliary functions'
    S = build_node_alias(G)
    print '2 creating random walks'
    create_rand_walks(S, num_paths, length_path, rw_filename)
    print '3 learning word2vec models using C code'
    comman = './word2vec -train '+rw_filename+' -output '+emb_filename+' -size '+str(emb_dim)+' -window '+str(winsize)+' -negative 5 -cbow 0 -min-count 0 -iter 5 -sample 1e-1'
    os.system(comman)
    model_w2v = w2v.Word2Vec.load_word2vec_format(emb_filename, binary=False)
    print '4 saving learned embeddings'
    model_w2v.save_word2vec_format(emb_filename)
    return model_w2v

def SBM_learn_deepwalk_3(G, num_paths, length_path, emb_dim, rw_filename, emb_filename, winsize):
    """
    learning SBM model through deepwalk, using gensim package
    saving all the sentences in memory, saves a lot of file I/O time
    can achieve 3x speed up compare to File I/O approach
    can not scale to very large networks

    Inputs:
    G: graph
    num_paths: number of random walks starting from each node
    length_path: length of each random walk
    rw_filename: file name to store the created corpus of sentences from graph G
    emb_filename: file name to store the learned embeddings of the nodes
    emb_dim: the dimensionality of the embeddings
    """
    print '1 building alias auxiliary functions'
    S = build_node_alias(G)
    print '2 creating random walks'
    sentence = create_rand_walks_inmem(S, num_paths, length_path)
    print '3 learning word2vec models'
#    sentence = w2v.LineSentence(rw_filename)
    model_w2v = w2v.Word2Vec(sentence, size=emb_dim, window=winsize, min_count=0, sg = 1, negative = 5, sample = 1e-1, workers=4)
#    print '4 saving learned embeddings'
#    model_w2v.save_word2vec_format(emb_filename)
    return model_w2v

#%%

def SBM_visual_tsne(labels, X):
    import tsne
    import pylab as Plot
    Y=tsne.tsne(X, 2)
    Plot.figure()
    Plot.scatter(Y[:,0], Y[:,1], 20, labels);
    Plot.show();
    return Y

def save_clusters_in_parallel(y, y_est,filename):
    """
    helper function to save the learned clustering results
    y - ground truth labels
    y_est - learned labels
    filename - to save results
    """
    f=open(filename, 'w')
    for i in range(len(y)):
        f.write(str(y[i]) + ','+str(y_est[i])+'\n')
    f.close
    return 1

def cal_metrics(labels, y_est_full):
    N = len(labels)
    acc_nmi = metrics.normalized_mutual_info_score(labels, y_est_full)
#    return acc_nmi
    Conf = metrics.confusion_matrix(labels, y_est_full)
    r,c = scipy.optimize.linear_sum_assignment(-1*Conf)
    acc_ccr = float(Conf[r,c].sum())/float(N)
    return acc_nmi, acc_ccr

def cal_modularity(G, nodelist, y):
    m = G.size()
    m = float(m)
    Q = 0.0
    n = len(nodelist)
    k=[]
    for e in nodelist:
        k.append(float(len(G[e])))
    for i in range(n):
        for j in range(i+1,n):
            if y[i] == y[j]:
                if G.has_edge(nodelist[i], nodelist[j]):
                    A = 1.0 - k[i]*k[j]/m
                else:
                    A = -1*k[i]*k[j]/m
                Q +=A
    return 2*Q/m

def cal_metrics_3(labels, y_est_full):
    N = len(labels)
    acc_nmi = metrics.normalized_mutual_info_score(labels, y_est_full)
    acc_ars = metrics.adjusted_rand_score(labels, y_est_full)
    Conf = metrics.confusion_matrix(labels, y_est_full)
    r,c = scipy.optimize.linear_sum_assignment(-1*Conf)
    acc_ccr = float(Conf[r,c].sum())/float(N)
    return acc_nmi, acc_ccr, acc_ars

def update_a_res(arry, acc, alg, param, value, i):
    if alg not in arry:
        arry[alg] = {}
    key = param + '- ' + str(value)
    if key not in arry[alg]:
        arry[alg][key]={}
    arry[alg][key][i] = acc
    return arry

def summary_res(nmi_arry, ccr_arry, ars_arry, truelabel, label, alg, param, value, i):
    # alg: 'deep'/'sc'/'abp',
    # param: 'c'/'N',
    # value: the value of param,
    # i: the random iter
    nmi, ccr, ars = cal_metrics_3(truelabel, label)
    print 'the NMI is:', nmi
    print 'the CCR is:', ccr
    nmi_arry = update_a_res(nmi_arry, nmi, alg, param, value, i)
    ccr_arry = update_a_res(ccr_arry, ccr, alg, param, value, i)
    ars_arry = update_a_res(ars_arry, ars, alg, param, value, i)
    return nmi_arry, ccr_arry, ars_arry

def plot_res_3(res):
    import matplotlib.pyplot as plt
    nmi = res[0]
    ccr = res[1]
#    ars = res[2]
    tm = nmi['deep'].keys()
    param = tm[0].split('-')[0]
    x_array = [float(z.split('-')[1].strip()) for z in tm ]
    x_array = sorted(x_array)
    tm = [param + '- ' + str(v) for v in x_array]
    # get nmi for three algs, mean and std
    nmi_deep_mean = [np.mean(nmi['deep'][z].values()) for z in tm]
    nmi_sc_mean = [np.mean(nmi['sc'][z].values()) for z in tm]
    nmi_abp_mean = [np.mean(nmi['abp'][z].values()) for z in tm]
    nmi_deep_std = [np.std(nmi['deep'][z].values()) for z in tm]
    nmi_sc_std = [np.std(nmi['sc'][z].values()) for z in tm]
    nmi_abp_std = [np.std(nmi['abp'][z].values()) for z in tm]
    # get ccr for three algs
    ccr_deep_mean = [np.mean(ccr['deep'][z].values()) for z in tm]
    ccr_sc_mean = [np.mean(ccr['sc'][z].values()) for z in tm]
    ccr_abp_mean = [np.mean(ccr['abp'][z].values()) for z in tm]
    ccr_deep_std = [np.std(ccr['deep'][z].values()) for z in tm]
    ccr_sc_std = [np.std(ccr['sc'][z].values()) for z in tm]
    ccr_abp_std = [np.std(ccr['abp'][z].values()) for z in tm]
    # plot
    # x - ccr, o - nmi
    # b- - deep, r-- - sc, g-. - adp
    plt.figure(1)
    plt.errorbar(x_array, nmi_deep_mean, yerr=nmi_deep_std, fmt='bo-')
    plt.errorbar(x_array, nmi_sc_mean, yerr=nmi_sc_std, fmt='ro--')
    plt.errorbar(x_array, nmi_abp_mean, yerr=nmi_abp_std, fmt='go-.')

    plt.errorbar(x_array, ccr_deep_mean, yerr=ccr_deep_std, fmt='bx-')
    plt.errorbar(x_array, ccr_sc_mean, yerr=ccr_sc_std, fmt='rx--')
    plt.errorbar(x_array, ccr_abp_mean, yerr=ccr_abp_std, fmt='gx-.')

    plt.legend(['NMI-New', 'NMI-SC', 'NMI-ABP', 'CCR-New', 'CCR-SC', 'CCR-ABP'], loc=0)
    plt.xlabel('Performance as function of '+param)

    plt.show()
    return x_array
    
def plot_res(res):
    # res can be nmi/ccr data structure, 
    # mt is the name of metric
    for alg in res:
        for para in res[alg]:
            u = res[alg][para].values()
            res[alg][para]['mean']=np.mean(u)
            res[alg][para]['std']=np.std(u)
    return res

#%%

def parse_txt_data(edgename, nodename):
    """
    additional helper fucntion for parsing graph data
    in txt files by Christy
    edgename: edge files,  each line: u,v  (node ids of a edge)
    nodename: node file, each line: nodeid, node-labels
    """
    model_params = {}
    G = nx.Graph()
    # add edges
    f = open(edgename, 'r')
    for l in f:
        c = l.split()
        G.add_edge(c[0],c[1], weight = 1.0)
    f.close()
    # add nodes
    g = open(nodename, 'r')
    for l in g:
        c = l.split()
        G.node[c[0]]['community'] = c[1]
    g.close()
    # get overall graph info
    nds = G.nodes()
    model_params['N'] = len(nds)
    labels = [G.node[i]['community'] for i in nds]
    uK = set(labels)
    model_params['K'] = len(uK)
    return G, model_params

def get_true_labels(G):
    """
    only for Christy's format
    """
    nodeslist = G.nodes()
    labels = [G.node[i]['community'] for i in nodeslist]
    ln = [int(t) for t in labels]
    return ln
