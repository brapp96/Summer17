function s = random_walk_weighted(curr,length,P,doNBT)
% Runs a random walk on positive values of G of length 'length' starting at
% 'curr'. If doNBT is set to 1, the function will ensure the random walk
% doesn't backtrack either. P contains the aliasing array to make
% generation of next step faster.
% Note: Abandoned for now since all graphs for the time being are
% unweighted, so aliasing is unnecessary.
s = zeros(1,length);
prev = -1;
for i = 1:length
    s(i) = curr;
    alias = P{s(i)};
    if isempty(alias)
        s = [];
        return
    end
    if doNBT
        x = find(prev==alias(:,1),1);
        if ~isempty(x)
            alias = alias([1:x-1,x+1:end],:);
            if isempty(alias)
                s(i+1:end) = [];
                return
            end
        end
        prev = curr;
    end
    x = ceil(rand()*size(alias,1));
    prob = alias(x,3);
    if rand() > prob
        curr = alias(x,2);
    else
        curr = alias(x,1);        
    end
end
end
