function [G,L] = import_graph_by_edges(N,K,c,lambda,iter)
filename = sprintf('N%d-K%d-c%.1f-la%.1f-iter%d.txt',N,K,c,lambda,iter);
fp = fopen(filename,'rb');
for i = 1:3 % strip comments
    fgets(fp);
end
indi = [];
indj = [];
for i = 1:N
    line = fgets(fp);
    I = eval(['[' line ']']);
    indi(end+1:end+numel(I)-1) = I(1)*ones(1,numel(I)-1);
    indj(end+1:end+numel(I)-1) = I(2:end);
end
G = sparse(indi+1,indj+1,1,N,N);
G = G + G';
L = eval(fgets(fp))+1;
fclose(fp);
end