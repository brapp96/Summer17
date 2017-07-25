function [Em_true,ccr,nmi] = node_embed_file(G,L,doNBT)
% Implements the node embeddings 'vec' algorithm of Ding et al. including a
% non-backtracking random walks option. This version works with file I/O
% and the use of the compiled "word2vec" code of Mikolov et al. 
% Inputs:
% G: graph in adjacency matrix form. Directed graphs should work; weighted
%    graphs should be tested but might work (TODO).
% L: ground truth of the embedding.
% doNBT: whether or not to do non-backtracking random walks on the graph.
% other parameters can be set manually for now; it's more annoying than
% helpful to have 10 different arguments to the program.
% Outputs:
% Em_true: the embedding produced by the algorithm.
% ccr: the correct classification rate.
% nmi: the normalized mutual information of the output.
% 
% Brian Rappaport, 7/24/17

% set vars
n = numel(L); % number of points
k = max(L); % number of communities
len = 60; % length of random walk
rw_reps = 10; % number of random walks per data point
dim = 50; % embedded dimension
winsize = 8; % window size
read_fp = 'sentences.txt';
write_fp = 'embeddings.txt';
numWorkers = 4; % honestly not really sure what this does but it's probably important

% write random walks to file
disp('creating random walks...');
nodes2file(G,read_fp,rw_reps,len,doNBT);
% run word2vec with external C code
disp('running embedding...');
command = ['./word2vec -train ' read_fp ' -output ' write_fp ...
          ' -size ' int2str(dim) ' -window ' int2str(winsize) ...
          ' -negative 5 -cbow 0 -sample 1e-4 -debug 2 -workers ' int2str(numWorkers)];
system(command);
% get embeddings from word2vec
U = file2embs(write_fp);
% run kmeans
Em = kmeans(U,k);
Em_true = get_true_emb(Em,L);
ccr = sum(Em_true == L)*100/n;
nmi = get_nmi(Em_true, L);
disp(ccr);
disp(nmi);
end

function U = file2embs(filename)
% Reads a file from word2vec and returns it as an array in memory.
fp = fopen(filename,'rb');
dims = fscanf(fp,'%d %d');
dims(1) = dims(1) - 1; % word2vec finds '</s>' as a word
U = zeros(dims');
fgets(fp);
for i = 1:dims(1)
    line = fgets(fp);
    [node,line] = strtok(line); %#ok<STTOK>
    U(str2double(node),:) = eval(['[' line ']']);
end
fclose(fp);
end

function nodes2file(G,filename,rw_reps,len,doNBT)
% Runs random walks on G and writes to a file.

% set variables
n = size(G,1);
P = create_aliases(G);
fp = fopen(filename,'wb');
% create walks
parfor i = 1:n
    for j = 1:rw_reps
        rw = random_walk(i,len,P,doNBT);
        for k = 1:numel(rw)
            fprintf(fp,'%d ',rw(k));
        end
        fprintf(fp,'\n');
    end
end
fclose(fp);
end
