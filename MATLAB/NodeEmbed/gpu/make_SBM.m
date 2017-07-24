function [G, P] = make_SBM(n,k,scaling_type,c,lambda)
% n is the number of nodes
% k is the number of communities
% scaling_type is either constant or logarithmic
% c is the constant fraction associated with the scaling
% lambda is the reduction percentage
% Brian Rappaport, 7/6/17

cluster_seq = (1:k);
P = repmat(cluster_seq, ceil(n/k),1);
P = P(1:n)';
if strcmp(scaling_type,'const')
    % odds if in same community is c/n, else is c(1-lambda)/n
    q = c*(1-lambda)/n;
elseif strcmp(scaling_type,'log')
    % odds if in same community is clog(n)/n, else is c(1-lambda)log(n)/n
    q = c*(1-lambda)*log(n)/n;
else
    error('scaling type must be ''const'' or ''log''');
end

% Build graph
indI = zeros(1,0);
indJ = zeros(1,0);
for i = 1:n
    I = rand([1,n]);
    ind = P(i)== P;
    I(ind) = I(ind)*(1-lambda);
    f = find(I<q);
    indI = [indI, f];
    indJ = [indJ, i*ones(1,numel(f))];
end
G = sparse(indI,indJ,ones(numel(indI),1), n, n);
G = max(G,G');


end

