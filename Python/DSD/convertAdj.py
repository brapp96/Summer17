## Convert PPI data files to Adjacency matrix + Protein name list
##
## 7/26/2017, Anuththari Gamage

import PPIparser
import numpy as np

ppbAdj, names = PPIparser.GetAdj('test.tab', -1)
np.savetxt('largePPIgraph.txt', ppbAdj, fmt='%d')

labels = np.array(names.keys())
np.savetxt('largePPIlabels.txt', labels, fmt='%s')
