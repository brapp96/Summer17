function [G,L] = import_graph_by_edges(N,K,c)
%filename = sprintf('more_graphs/N%d-K%d-c%.1f.txt',N,K,c);
filename = 'graphs/PPIadj.txt';
fp = fopen(filename,'rb');
 for i = 1:3 % strip comments
     fgets(fp);
 end
indi = [];
indj = [];
N = 0;
while (~feof(fp))
    line = fgets(fp);
    N = N+1;
    I = eval(['[' line ']']);
    indi(end+1:end+numel(I)-1) = I(1)*ones(1,numel(I)-1);
    indj(end+1:end+numel(I)-1) = I(2:end);
end
G = sparse(indi+1,indj+1,1,N,N);
G = G + G';
L = 0;
%L = eval(fgets(fp))+1;
fclose(fp);
end
