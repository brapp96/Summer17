import PPIparser
import numpy as np
import pdb
import networkx as nx

filename = 'PPIdata.txt';
(ppbAdj, names) = PPIparser.GetAdj(filename,0);

nameArray = np.array(names.keys())
np.savetxt('PPInames.txt',nameArray,'%s')

graph = nx.from_numpy_matrix(ppbAdj)
nx.write_adjlist(graph, 'PPIadj.txt')

#pdb.set_trace()


