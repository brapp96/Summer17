function [Em_true,ccr_val,nmi_val] = node_embed_file(G,labels,doNBT)
% set vars
n = numel(labels);
k = max(labels);
len = 60; % length of random walk
rw_reps = 10; % number of random walks per data point
dim = 50; % embedded dimension
winsize = 8; % window size
read_fp = 'sentences.txt';
write_fp = 'embeddings.txt';
numWorkers = 4;

% write random walks to file
nodes2file(G,read_fp,rw_reps,len,doNBT);
% run word2vec
command = ['./word2vec -train ' read_fp ' -output ' write_fp ...
          ' -size ' int2str(dim) ' -window ' int2str(winsize) ...
          ' -negative 5 -cbow 0 -sample 1e-4 -workers ' int2str(numWorkers)];
system(command);
% get embeddings from word2vec
U = file2embs(write_fp);
% run kmeans
Em = kmeans(U,k);
Em_true = get_true_emb(Em,labels);
ccr_val = sum(Em_true == labels')*100/n;
nmi_val = nmi(labels', Em_true);
end

function U = file2embs(filename)
% U is returned in order 1-n
fp = fopen(filename,'rb');
dims = fscanf(fp,'%d %d');
dims(1) = dims(1) - 1;
U = zeros(dims');
fgets(fp);
for i = 1:dims(1)
    line = fgets(fp);
    [node,line] = strtok(line);
    U(str2double(node),:) = str2num(line);
end
fclose(fp);
end

function nodes2file(G,filename,rw_reps,len,doNBT)
% set variables
n = size(G,1);
P = create_aliases(G);

fp = fopen(filename,'wb');
% create walks
for i = 1:n
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