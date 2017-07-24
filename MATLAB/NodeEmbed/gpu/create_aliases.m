function P = create_aliases(G)
% Initializes aliasing array for fast choice from discrete random
% distribution
n = size(G,1);
P = cell(1,n);

[gi, gj, gv] = find(G);

for i = 1:n
    Ginds = gj(gi == i);
    I = gv(gi == i);
    k = numel(I);
    if k == 0, continue; end
    list = zeros(k,3);
    [I,inds] = sort(I/sum(I)*k);
    loc = find(I>1,1);
    if isempty(loc), loc = 1; end
    list(1:loc-1,1) = Ginds(inds(1:loc-1));
    list(1:loc-1,3) = I(1:loc-1);
    ind = 1;
    for f = loc:k
        while I(f) > 1
            list(ind,2) = Ginds(inds(f));
            I(f) = I(f) - (1 - list(ind,3));
            ind = ind + 1;
        end
        list(f,1) = Ginds(inds(f));
        list(f,3) = I(f);
    end
 P{i} = list;
end
end
