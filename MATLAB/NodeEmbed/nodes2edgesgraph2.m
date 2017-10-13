function E = nodes2edgesgraph2(G,doNBT)
profile on;
G = triu(G,1)+triu(G,1)';
n = size(G,1);
row = 0;
indi = [];
indj = [];
inds = {n};
bds = ones(n+1,1);
startinds = 1;
for i = 1:n
    inds{i} = find(G(i,:));
    bds(i+1) = numel(inds{i}) + bds(i);
end
for i = 1:n
    is = inds{i};
    for j = 1:numel(is)
        row = row + 1;
        js = bds(is(j)):bds(is(j)+1)-1;
        if doNBT
            js(j) = [];
        end
        endinds = startinds + js;
        indi(startinds:endinds) = row;
        indj(startinds:endinds) = js;
        startinds = endinds + 1;
    end
end
num_elems = sum(sum(G));
E = sparse(indi,indj,1,num_elems,num_elems);
profile viewer;
end