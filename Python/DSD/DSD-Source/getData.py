import PPIparser
import numpy as np
import pdb
import networkx as nx
import sklearn.cluster as sc
 
filename = 'PPIdata.txt';
(ppbAdj, names) = PPIparser.GetAdj(filename,0);

nameArray = np.array(names.keys());
indexArray = np.array(names.values());
pdb.set_trace()
#
#graph = nx.from_numpy_matrix(ppbAdj);
#nx.write_adjlist(graph, 'PPIadj.txt')
#
#N = 5885
#DSD = np.loadtxt('results.DSD1', skiprows=1, usecols=range(1,N));
#pdb.set_trace()
#DSD_sim = 1/(DSD + np.eye(DSD.shape[0]))
#
#            
## Apply spectral clustering and reorder labels using Hungarian algorithm
#print('Applying spectral clustering...')
#labels = sc.spectral_clustering(DSD_sim, n_clusters=2)
#pdb.set_trace()
##Conf = metrics.confusion_matrix(true_labels, labels)
##row, col = scipy.optimize.linear_sum_assignment(-1*Conf)
##
### Get metrics
##print('Calculating metrics...')
##acc_nmi[i] = metrics.normalized_mutual_info_score(true_labels,labels)
##acc_ccr[i] = float(Conf[row, col].sum())/float(N)
            

