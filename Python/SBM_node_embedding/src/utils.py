"""
"""

import numpy as np

import VEClib as algs
    
# initialize the stochastic block models with different parameter settings:        
def SBM_param_init_n_unequalcomm(K, N, lambda_n, alpha_n, r=0.5):
    """
    create SBM model parameters
    community weights are balanced = [r, 1-r]
    """
    SBM_params = {}
    if K>2:
        print 'this function cannot support K>2, warning'
    SBM_params['K'] = K
    SBM_params['N'] = N
    SBM_params['lambda_n'] = float(lambda_n)
    SBM_params['alpha'] = alpha_n/float(N)
    SBM_params['B0'] = lambda_n * np.eye(K) + (1-lambda_n)*np.ones((K,K)) 
    z = [r, 1.0-r]
    SBM_params['a'] = z
    return SBM_params

# initialize the stochastic block models with different parameter settings:        
def SBM_param_init_n_badQ(K, N, lambda_n, alpha_n, qr=1.0):
    """
    create SBM model parameters
    community weights are balanced = [1/k,...1/k]
    """
    SBM_params = {}
    SBM_params['K'] = K
    SBM_params['N'] = N
    SBM_params['lambda_n'] = float(lambda_n)
    SBM_params['alpha'] = alpha_n/float(N)
    SBM_params['B0'] = lambda_n * np.eye(K) + (1-lambda_n)*np.ones((K,K)) 
    SBM_params['B0'][1][1] = qr
    z = np.ones((1,K))
    SBM_params['a'] = z[0]/float(K)
    return SBM_params
            
#%%    
"""
updated 11-23-16
The following sub-block is a quick by-pass to the LDA approach for clustering
NOTE: the current by-pass uses the fast Gibbs Sampling approach
NOTE: the current implementation imported the bag-of-words model into 
"""
from sklearn.feature_extraction.text import CountVectorizer
import lda
def SBM_learn_deepwalk_lda(G, num_paths, length_path, emb_dim, rw_filename, emb_filename):
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
    print '1 building alias auluxy functions'
    S = algs.build_node_alias(G)    
    print '2 creating random walks'
    algs.create_rand_walks(S, num_paths, length_path, rw_filename)
    print '3 learning lda models'
    q = []
    f = open(rw_filename,'rb')
    for l in f:
        q.append(l.strip())
    vec = CountVectorizer(min_df=1)
    c = vec.fit_transform(q)
    names = vec.get_feature_names() # BoW feature names = nodes
    del q  # do this to make sure it minimizes the 
    model = lda.LDA(n_topics=emb_dim, n_iter=1000, random_state=1)
    model.fit(c)
    m = model.topic_word_
    # now lets construct a dictionary first:
    model_w2v = {}
    for i in range(len(names)):
        key = str(names[i])
        vector = m[:,i]
        model_w2v[key]=vector
    return model_w2v 
    
def SBM_learn_deepwalk_lda_another(G, num_paths, length_path, emb_dim, rw_filename, emb_filename):
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
    print '1 building alias auluxy functions'
    S = algs.build_node_alias(G)    
    print '2 creating random walks'
    algs.create_rand_walks(S, num_paths, length_path, rw_filename)
    print '3 learning lda models'
    q = []
    f = open(rw_filename,'rb')
    for l in f:
        q.append(l.strip())
    return q
#    vec = CountVectorizer(min_df=1)
#    c = vec.fit_transform(q)
#    names = vec.get_feature_names() # BoW feature names = nodes
#    del q  # do this to make sure it minimizes the 
#    model = lda.LDA(n_topics=emb_dim, n_iter=1000, random_state=1)
#    model.fit(c)
#    m = model.topic_word_
#    # now lets construct a dictionary first:
#    model_w2v = {}
#    for i in range(len(names)):
#        key = str(names[i])
#        vector = m[:,i]
#        model_w2v[key]=vector
#    return model_w2v 
    
    
def lda_to_mat(ldamodel, nodelist, emb_dim):
    # util functions to put dictionary model into matrices, row-wise indexed
    N = len(nodelist)
    X = np.zeros((N, emb_dim))
    for i in xrange(N):
        keyid = nodelist[i]
        if keyid in ldamodel:
            X[i,:] = ldamodel[keyid]
        else:
            print 'node', keyid, 'is missing'
    return X
def lda_to_mat_deepwalkdata(ldamodel, N, emb_dim):
    X = np.zeros((N, emb_dim))
    for i in xrange(N):
        keyid = str(i)+'10'
        if keyid in ldamodel:
            X[i,:] = ldamodel[keyid]
        else:
            print 'node', keyid, 'is missing'
    return X
def lda_normalize_embs(mat, option = 0):
    newmat = np.zeros(mat.shape)
    ## row-norms
    if option ==0:
        print 'option 0, no normalization'
        newmat = np.copy(mat)
    if option ==2:
        print 'option 2, l2 normalization'
        norms = np.linalg.norm(mat,  axis = 1)
        for i in range(len(mat)):
            newmat[i,:] = mat[i,:]/norms[i]
    if option ==1:
        print 'option 1, l1 normalization'
        norms = np.sum(mat, axis = 1)
        for i in range(len(mat)):
            newmat[i,:] = mat[i,:]/norms[i]
    return newmat
    
def clustering_embs(mat, K):
    ## apply k-means clustering algorithm to get labels
    from sklearn.cluster import KMeans
    k_means2 = KMeans(n_clusters=K, max_iter=100, precompute_distances=False)
    k_means2.fit(mat)
    y_hat = k_means2.labels_
    return y_hat    
    
def clustering_embs_noramlized(mat, K, option =0):
    ## apply k-means clustering algorithm to get labels
    from sklearn.cluster import KMeans
    ## row-norms, try L2 norm first
    if option ==0:
        print 'option 0, no normalization'
    if option ==2:
        print 'option 2, l2 normalization'
        norms = np.linalg.norm(mat,  axis = 1)
        for i in range(len(mat)):
            mat[i,:] = mat[i,:]/norms[i]
    if option ==1:
        print 'option 1, l1 normalization'
        norms = np.sum(mat, axis = 1)
        for i in range(len(mat)):
            mat[i,:] = mat[i,:]/norms[i]
    ##
    k_means2 = KMeans(n_clusters=K, max_iter=100, precompute_distances=False)
    k_means2.fit(mat)
    y_hat = k_means2.labels_
    return y_hat    

def maxfinding_embs_noramlized(mat, K, option =1):
    ## simply go and find the maximum
    ## row-norms, try L2 norm first
    if option ==0:
        print 'option 0, no normalization'
    if option ==2:
        print 'option 2, l2 normalization'
        norms = np.linalg.norm(mat,  axis = 1)
        for i in range(len(mat)):
            mat[i,:] = mat[i,:]/norms[i]
    if option ==1:
        print 'option 1, l1 normalization'
        norms = np.sum(mat, axis = 1)
        for i in range(len(mat)):
            mat[i,:] = mat[i,:]/norms[i]
    ##%%
    y_hat = []
    for i in range(len(mat)):
        y_hat.append(np.argmax(mat[i,:]))
#    k_means2 = KMeans(n_clusters=K, max_iter=100, precompute_distances=False)
#    k_means2.fit(mat)
#    y_hat = k_means2.labels_
    return y_hat  
