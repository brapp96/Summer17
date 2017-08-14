function E = nodes2edgesgraph(G,doNBT)
G = triu(G,1)+triu(G,1)';
n = size(G,1);
row = 0;
indi = [];
indj = [];
for i = 1:n
    for j = 1:n
        if j == i
            continue;
        end
        row = row + 1;
        % from i to j
        inds = find(G(j,:));
        if doNBT
            inds(inds==i) = [];
        end
        inds(inds>=j) = inds(inds>=j) - 1;
        inds = inds + (n-1)*(j-1);
        indi(end+1:end+numel(inds)) = row*ones(1,numel(inds));
        indj(end+1:end+numel(inds)) = inds;
    end
end
E = sparse(indi,indj,1,n*(n-1),n*(n-1));
end