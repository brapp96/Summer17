function s = random_walk(curr,length,G,doNBT)
% Runs a random walk on positive values of G of length 'length' starting at
% 'curr'. If doNBT is set to 1, the function will ensure the random walk
% doesn't backtrack. This version only works for unweighted graphs but is
% correspondingly faster. Also note that here we've set the walks to be
% begrudgingly backtracking ones, not entirely non-backtracking.

s = zeros(1,length);
prev = -1;
if isempty(find(G(:,curr),1))
    s = [];
    return
end
for i = 1:length
    s(i) = curr;
    next = find(G(:,s(i)));
    if doNBT
        next(next==prev) = [];
        if isempty(next)
            next = prev;
        end
        prev = curr;
    end
    curr = next(ceil(rand()*numel(next)));
end
end
