# Node Embedding through Non-Backtracking Random Walks

This is the github repository containing the code referred to in "Faster Clustering Using Non-Backtracking Random Walks", available on arXiv. To use this code yourself, you need to have a copy of MATLAB and the Parallel Computing Toolbox. You can run this from the command line using "testfile", which you can modify to your desired purpose, or you can use the function directly, "node\_embed\_file(G,L,doNBT,len)", where G is a graph in sparse adjacency matrix form, L is the ground truth of the labels, doNBT determines whether or not the random walks are backtracking, and len determines the length of the random walks. More parameters can be adjusted within the file.

Also necessary is a compiled word2vec binary, which is included in this repository as well. Necessary citations are included in the bibliography of the paper.


For commercial use of this code, please cite the following.
Brian Rappaport, Anuththari Gamage, and Shuchin Aeron. "Faster Clustering Using Non-Backtracking Random Walks". In: arXiv (2017).
