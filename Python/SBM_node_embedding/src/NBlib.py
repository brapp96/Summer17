"""

Non-backtracking random walks library

Taken largely from Node Embeddings paper by Ding et al.
Modified to allow non-backtracking random walks by Brian Rappaport, 2017.


"""
import numpy as np

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
    K  = len(J)
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
    if inMem:
        fwrite = open(filename,'w')
    else:
        sentence = []

    nodes = S.keys()
    for nd in nodes:
        for i in range(num_paths):
            walk = [nd] # start as nd
            cur = -1
            for j in range(length_path):
                prev = cur                
                cur = walk[-1]
                next_nds = list(S[cur]['names']) # need full copy to pass value       
                if len(next_nds)<1:
                    break
                J = S[cur]['J']
                q = S[cur]['q']
                if NBT and prev in next_nds:
                    if len(next_nds)==1:
                        break
                    ind = next_nds.index(prev)
                    del next_nds[ind]
                    J = np.delete(J,ind)
                    q = np.delete(q,ind)
                rd = alias_draw(J,q)
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
