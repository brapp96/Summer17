import PPIparser
import numpy as np
import pdb
import networkx as nx
import sklearn.cluster as sc
 
#filename = 'PPIdata.txt';
#(ppbAdj, names) = PPIparser.GetAdj(filename,0);
#
#nameArray = np.array(names.keys());
#indexArray = np.array(names.values());
#pdb.set_trace()

#graph = nx.from_numpy_matrix(ppbAdj);
#nx.write_adjlist(graph, 'PPIadj.txt')

N = 5885
DSD = np.loadtxt('results.DSD1', skiprows=1, usecols=range(1,N));
#pdb.set_trace()
DSD_sim = 1/(DSD + np.eye(DSD.shape[0]))
#pdb.set_trace()
            
# Apply spectral clustering and reorder labels using Hungarian algorithm
print('Applying spectral clustering...')
labels = sc.spectral_clustering(DSD_sim, n_clusters=43)
np.savetxt('results_DSD43.txt', labels, fmt='%d')

totals = np.zeros(43)
for i in range(0,43):
    totals[i] = np.sum(labels == i)

print(np.sort(totals))

max_size = np.max(totals)
max_cluster = np.where(totals == max_size)
max_cluster_label = max_cluster[0][0]

proteinNames = np.loadtxt('proteinNames_DSD.txt', dtype='S6')
in_LC = np.where(labels == max_cluster_label) 
pdb.set_trace()
proteins_in_LC = proteinNames[in_LC]
np.savetxt('largestCluster2.txt', proteins_in_LC, fmt='%s')
#pdb.set_trace()
#pdb.set_trace()
#Conf = metrics.confusion_matrix(true_labels, labels)
#row, col = scipy.optimize.linear_sum_assignment(-1*Conf)
#
## Get metrics
#print('Calculating metrics...')
#acc_nmi[i] = metrics.normalized_mutual_info_score(true_labels,labels)
#acc_ccr[i] = float(Conf[row, col].sum())/float(N)
            

