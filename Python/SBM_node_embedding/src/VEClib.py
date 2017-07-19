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

import pickle
import networkx as nx
from sklearn import metrics
import gensim.models.word2vec as w2v
import numpy as np
import scipy
from . import SBMlib as SBM

if __name__ == '__main__':
    exec('/home/brian/Documents/summer17/Python/SBM_node_embeddings/NBtest.py')

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
        J, q = SBM.alias_setup(entry['weights'])
        entry['J'] = J
        entry['q'] = q
        nodes_rw[nd] = entry
    return nodes_rw

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
    K = len(J)
    # Draw from the overall uniform mixture.
    kk = int(np.floor(np.random.rand()*K))
    # Draw from the binary mixture, either keeping the
    # small one, or choosing the associated larger one.
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]

def create_random_walks(S, num_paths, length_path, filename, inMem=False, NBT=False):
    """
    S = from the build_node_alias
    filename - where to write results
    using exp(rating) as edge weight
    """
    sentence = []
    if ~inMem:
        fwrite = open(filename, 'w')
    nodes = S.keys()
    for nd in nodes:
        for _ in range(num_paths):
            walk = [nd] # start as nd
            cur = -1
            for _ in range(length_path):
                prev = cur
                cur = walk[-1]
                next_nds = list(S[cur]['names']) # need full copy to pass value
                if len(next_nds) < 1:
                    break
                J = S[cur]['J']
                q = S[cur]['q']
                if NBT and prev in next_nds:
                    if len(next_nds) == 1:
                        break
                    ind = next_nds.index(prev)
                    del next_nds[ind]
                    J = np.delete(J, ind)
                    q = np.delete(q, ind)
                rd = alias_draw(J, q)
                nextnd = next_nds[rd]
                walk.append(nextnd)
            walk = [str(x) for x in walk]
            if inMem:
                sentence.append(walk)
            else:
                fwrite.write(" ".join(walk) + '\n')
    if ~inMem:
        fwrite.close()
    return sentence

#%%
def SBM_learn_deepwalk(G, rw_filename, emb_filename, num_paths=10, length_path=60, emb_dim=50,
                       winsize=8, neg_samples=5, NBT=False, speedup=True, inMem=False, save=False):
    """
    learning SBM model through deepwalk, using gensim package
    Inputs:
    G: graph
    num_paths: number of random walks starting from each node
    length_path: length of each random walk
    rw_filename: file name to store the created corpus of sentences from graph G
    emb_filename: file name to store the learned embeddings of the nodes
    emb_dim: the dimensionality of the embeddings
    winsize: size of the window used
    NBT: whether to use non backtracking random walks
    speedup: whether to use the C code to accelerate the program
    inMem: whether to work in memory or not (possible only for smaller datasets)
    save: whether to save the word2vec results to disk
    """
    print '1 building alias auxiliary functions'
    S = build_node_alias(G)
    print '2 creating random walks'
    sentence = create_random_walks(S, num_paths, length_path, rw_filename, inMem, NBT)
    print '3 learning word2vec models'
    if speedup:
        import os
        comman = './word2vec -train '+rw_filename+' -output '+emb_filename+' -size '+str(emb_dim) \
            +' -window '+str(winsize)+' -negative '+str(neg_samples)+' -cbow 0 -min-count 0 -iter 5 -sample 1e-1'
        os.system(comman)
        model_w2v = w2v.load_word2vec_format(emb_filename, binary=False)
    else:
        if ~inMem:
            sentence = w2v.LineSentence(rw_filename)
        model_w2v = w2v.Word2Vec(sentence, size=emb_dim, window=winsize, min_count=0,
                                 sg=1, negative=neg_samples, sample=1e-1, workers=5, iter=3)
    if save:
        print '4 saving learned embeddings'
        model_w2v.save_word2vec_format(emb_filename)
    return model_w2v

#%%
def SBM_visual_tsne(labels, X):
    from . import tsne
    import pylab as Plot
    Y = tsne.tsne(X, 2)
    Plot.figure()
    Plot.scatter(Y[:, 0], Y[:, 1], 20, labels)
    Plot.show()
    return Y

def save_clusters_in_parallel(y, y_est, filename):
    """
    helper function to save the learned clustering results
    y - ground truth labels
    y_est - learned labels
    filename - to save results
    """
    f = open(filename, 'w')
    for i in range(len(y)):
        f.write(str(y[i]) + ','+str(y_est[i])+'\n')
    f.close()
    return 1

def cal_metrics(labels, y_est_full):
    N = len(labels)
    acc_nmi = metrics.normalized_mutual_info_score(labels, y_est_full)
#    return acc_nmi
    Conf = metrics.confusion_matrix(labels, y_est_full)
    r, c = scipy.optimize.linear_sum_assignment(-1*Conf)
    acc_ccr = float(Conf[r, c].sum())/float(N)
    return acc_nmi, acc_ccr

def cal_modularity(G, nodelist, y):
    m = G.size()
    m = float(m)
    Q = 0.0
    n = len(nodelist)
    k = []
    for e in nodelist:
        k.append(float(len(G[e])))
    for i in range(n):
        for j in range(i+1, n):
            if y[i] == y[j]:
                if G.has_edge(nodelist[i], nodelist[j]):
                    A = 1.0 - k[i]*k[j]/m
                else:
                    A = -1*k[i]*k[j]/m
                Q += A
    return 2*Q/m

def cal_metrics_3(labels, y_est_full):
    N = len(labels)
    acc_nmi = metrics.normalized_mutual_info_score(labels, y_est_full)
    acc_ars = metrics.adjusted_rand_score(labels, y_est_full)
    Conf = metrics.confusion_matrix(labels, y_est_full)
    r, c = scipy.optimize.linear_sum_assignment(-1*Conf)
    acc_ccr = float(Conf[r, c].sum())/float(N)
    return acc_nmi, acc_ccr, acc_ars

def update_a_res(arry, acc, alg, param, value, i):
    if alg not in arry:
        arry[alg] = {}
    key = param + '- ' + str(value)
    if key not in arry[alg]:
        arry[alg][key] = {}
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
    x_array = [float(z.split('-')[1].strip()) for z in tm]
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
            res[alg][para]['mean'] = np.mean(u)
            res[alg][para]['std'] = np.std(u)
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
        G.add_edge(c[0], c[1], weight=1.0)
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

def get_label_list(G):
    # only work for simulated graphs
    nodeslist = G.nodes()
    ln = [G.node[i]['community'] for i in nodeslist]
    nodeslist = [str(x) for x in nodeslist]
    return ln, nodeslist

def plot_res(N, params):
    import matplotlib.pyplot as plt
    fstring = "exp1%d" % N
    res = pickle.load(open("%s.pkl" % fstring, 'rb'))
    nmi = res[0]
    ccr = res[1]
    tm = nmi[params[0]].keys()
    param = tm[0].split('-')[0]
    x_array = [float(z.split('-')[1].strip()) for z in tm]
    x_array = sorted(x_array)
    tm = [param + '- ' + str(v) for v in x_array]
    
    # get nmi and ccr for all algos 
    nmi_mean = np.array((1,size(z)))
    for p in params:
        nmi_mean[p] = [np.mean(nmi[p][z].values()) for z in tm]
        nmi_std[p] = [np.std(nmi[p][z].values()) for z in tm]
        ccr_mean[p] = [np.mean(ccr[p][z].values()) for z in tm]
        ccr_std[p] = [np.std(ccr[p][z].values()) for z in tm]
    
    plt.figure(1, figsize=(10, 6))
    for p in params:
        plt.errorbar(x_array, nmi_mean[p], yerr=nmi_std[p], markersize=8, linewidth=1.5)
        plt.errorbar(x_array, ccr_mean[p], yerr=ccr_std[p], markersize=8, linewidth=1.5)

    legend = ["nmi-%s" % p for p in params]
    legend = ["ccr-%s" % p for p in params]
    plt.legend(legend, loc=0)
    plt.xlabel(param)
    plt.xlim(x_array[0]-0.1, x_array[-1]+0.1)
    plt.ylim(-0.05, 1.05)
    plt.show()
    plt.savefig(fstring+'.eps', bbox_inches='tight', format='eps')
    plt.savefig(fstring+'.png', bbox_inches='tight', format='png')
