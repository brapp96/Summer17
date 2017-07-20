"""
This file contains all the functions for the node embedding algorithm.
"""

import os
import pickle
from sklearn import metrics
from gensim.models import word2vec as w2v, keyedvectors as kv
import numpy as np
import scipy
from . import SBMlib as SBM, globVars

def build_node_alias(G):
    """
    Builds a dictionary that can be used to generate random walks on G.
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
    Draw random samples from a discrete distribution with specific nonuniform
    weights. Code was adapted from the following source:
    https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
   """

    # Draw from the overall uniform mixture.
    kk = int(np.random.rand()*len(J))
    # Draw from the binary mixture.
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]

def create_random_walks(S, num_paths, length_path, filename,
                        inMem=False, NBT=False):
    """
    Create the list of random walk "sentences" on the graph using the adjacency
    list S from build_node_alias().
    """
    sentence = []
    if ~inMem:
        #FILEPATH gives results directory, which is where these text files are stored
        fp = open(globVars.FILEPATH+filename, 'w')
    nodes = S.keys()
    for nd in nodes:
        for _ in range(num_paths):
            walk = [nd] # start at current node
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
                    J = J[range(0, ind)+range(ind+1, len(J))]
                    q = q[range(0, ind)+range(ind+1, len(q))]
                rd = alias_draw(J, q)
                nextnd = next_nds[rd]
                walk.append(nextnd)
            walk = [str(x) for x in walk]
            if inMem:
                sentence.append(walk)
            else:
                fp.write(" ".join(walk) + '\n')
    if ~inMem:
        fp.close()
    return sentence

def SBM_learn_deepwalk(G, rw_filename, emb_filename, num_reps=10, length=60,
                       dim=50, winsize=8, NBT=False, useC=True, inMem=False):
    """
    Learn SBM model through random walks, using gensim package and original C
    code.
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
    w2vpath = globVars.FILEPATH+'../src/'
    globVars.printDebug('1 building alias auxiliary functions')
    S = build_node_alias(G)
    globVars.printDebug('2 creating random walks')
    sentence = create_random_walks(S, num_reps, length, rw_filename, inMem, NBT)
    globVars.printDebug('3 learning word2vec models')
    if useC:
        dbStatus = int(globVars.DEBUG)
        command = w2vpath+'word2vec -train '+globVars.FILEPATH+rw_filename\
                  +' -output '+globVars.FILEPATH+emb_filename+' -size '\
                  +str(dim)+' -window '+str(winsize)+' -negative 5 -cbow 0 '\
                  +'-iter 3 -sample 1e-4 -debug '+str(dbStatus) +' -workers 4'\
                  +' >> '+globVars.FILEPATH+'test.log'
        os.system(command)
    else:
        if ~inMem:
            sentence = w2v.LineSentence(rw_filename)
        model_w2v_calc = w2v.Word2Vec(sentence, size=dim, window=winsize,
                                      min_count=0, sg=1, negative=5,
                                      sample=1e-1, workers=5, iter=3)
        model_w2v_calc.save_word2vec_format(globVars.FILEPATH+emb_filename)
    model_w2v = kv.KeyedVectors.load_word2vec_format(globVars.FILEPATH+emb_filename)
    return model_w2v

#%%

def summary_res(nmi_arry, ccr_arry, ars_arry, gt, label, alg, param_str):
    """
    Top level metrics function: calculates various metrics and updates
    the metric arrays.
    """
    nmi, ccr, ars = cal_metrics(gt, label)
    globVars.printDebug('the NMI is: '+str(nmi)+'; the CCR is: '+str(ccr))
    for (val, array) in [[nmi, nmi_arry], [ccr, ccr_arry], [ars, ars_arry]]:
        array = update_metric_arrays(array, val, alg, param_str)
    return nmi_arry, ccr_arry, ars_arry

def cal_metrics(labels, y_est_full):
    """
    Calculates nmi, ccr, and ars for given predicted results and ground truth.
    """
    N = len(labels)
    acc_nmi = metrics.normalized_mutual_info_score(labels, y_est_full)
    acc_ars = metrics.adjusted_rand_score(labels, y_est_full)
    Conf = metrics.confusion_matrix(labels, y_est_full)
    r, c = scipy.optimize.linear_sum_assignment(-1*Conf)
    acc_ccr = float(Conf[r, c].sum())/float(N)
    return acc_nmi, acc_ccr, acc_ars

def update_metric_arrays(arry, acc, alg, param_str):
    """
    Updates the metric arrays as they are added to.
    """
    if alg not in arry:
        arry[alg] = {}
    key = param_str
    arry[alg][key] = acc
    return arry

def plot_res(N, params, fignum):
    """
    Plots the metrics for each type and for the parameter that is varied.
    TODO: this won't work right now; need to fix
    """
    import matplotlib.pyplot as plt
    fstring = 'N'+str(N)+';K'+str(K)+';c'+str(c)+';la'+str(lambda_n)+';iter'
    res = pickle.load(open("%s.pkl" % fstring, 'rb'))
    nmi = res[0]
    ccr = res[1]
    tm = nmi[params[0]].keys()
    param = tm[0].split('-')[0]
    x_array = [float(z.split('-')[1].strip()) for z in tm]
    x_array = sorted(x_array)
    tm = [param + '- ' + str(v) for v in x_array]
    # get nmi and ccr for all algos
    nmi_mean = {}
    nmi_std = {}
    ccr_mean = {}
    ccr_std = {}
    for p in params:
        nmi_mean[p] = [np.mean(nmi[p][z].values()) for z in tm]
        nmi_std[p] = [np.std(nmi[p][z].values()) for z in tm]
        ccr_mean[p] = [np.mean(ccr[p][z].values()) for z in tm]
        ccr_std[p] = [np.std(ccr[p][z].values()) for z in tm]
    fig = plt.figure(fignum, figsize=(10, 6))
    i = 0
    cmap = plt.get_cmap('cubehelix')
    for p in params:
        color = cmap(float(i)/len(params))
        plt.errorbar(x_array, nmi_mean[p], yerr=nmi_std[p],
                     color=color, markersize=8)
        plt.errorbar(x_array, ccr_mean[p], yerr=ccr_std[p],
                     color=color, markersize=8)
        i += 1
    legend = ["nmi-%s" % p for p in params]
    legend.extend(["ccr-%s" % p for p in params])
    plt.legend(legend, loc=0)
    plt.xlabel(param)
    plt.xlim(x_array[0]-0.1, x_array[-1]+0.1)
    plt.ylim(-0.05, 1.05)
    plt.savefig(fstring+'.eps', bbox_inches='tight', format='eps')
    plt.savefig(fstring+'.png', bbox_inches='tight', format='png')
    return fig

