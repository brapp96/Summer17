
function [G, labels] = make_SBM(n,k,scaling_type,c,lambda)                  % modified to return labels
% n is the number of nodes
% k is the number of communities
% scaling_type is either constant or logarithmic
% c is the constant fraction associated with the scaling
% lambda is the reduction percentage
% Brian Rappaport, 7/6/17

% We're using equal partitions
P = repmat(1:k,ceil(n/k),1);
P = P(1:n);
labels = P';
% get connections
if strcmp(scaling_type,'const')
    % odds if in same community is c/n, else is c(1-lambda)/n
    q = c*(1-lambda)/n;
elseif strcmp(scaling_type,'log')
    % odds if in same community is clog(n)/n, else is c(1-lambda)log(n)/n
    q = c*(1-lambda)*log(n)/n;
else
    error('scaling type must be ''const'' or ''log''');
end
% build graph
indI = [];
indJ = [];
for i = 1:n
    I = rand(1,n);
    ind = P(i)==P;
    I(ind) = I(ind)*(1-lambda);
    f = find(I<q);
    indI(end+1:end+numel(f)) = f;
    indJ(end+1:end+numel(f)) = i;
end
G = sparse(indI,indJ,ones(numel(indI),1),n,n,numel(indI));
G = max(G,G');
end