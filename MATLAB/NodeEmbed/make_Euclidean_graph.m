function G = make_Euclidean_graph(N,type,varargin)                              
% Creates the similarity graph. 
% N is an n x 2 matrix of points in R^2
% Copyright (c) 2012, Ingo Bork
% Copyright (c) 2003, Jochen Lenz
% Copyright (c) 2013, Oliver Woodford
% Edits 2017, Brian Rappaport

n = size(N,1);
G = zeros(n);
dist = @(x,y) exp(-norm(x-y));
switch type
    case 'full'
        for i = 2:n
            for j = 1:i-1
                G(i,j) = dist(N(i,:),N(j,:));
            end
        end
        G = G + G';
    case {'knear','k'}
        k = varargin{1} + 1; % to exclude zeros
        for i = 1:n
            dist = sqrt(sum((repmat(N(i,:),n,1) - N).^ 2,2));
            [s,O] = sort(dist,'ascend');
            indi(1,(i-1)*k+1:i*k) = i;
            indj(1,(i-1)*k+1:i*k) = O(1:k);
            inds(1,(i-1)*k+1:i*k) = s(1:k);
        end
        G = sparse(indi,indj,inds,n,n,numel(indi));
        G = max(G,G');
    case {'eps','e'}
        e = varargin{1};
        indi = [];
        indj = [];
        inds = [];
        for i = 1:n
            dist = sqrt(sum((repmat(N(i,:),n,1) - N).^ 2,2));
            f = find(dist<e);
            indi(end+1:end+numel(f)) = f;
            indj(end+1:end+numel(f)) = i;
            inds(end+1:end+numel(f)) = dist(f);
        end
        G = sparse(indi,indj,inds,n,n,numel(indi));
        % G will be symmetric by nature so the max step isn't necessary
    otherwise
        error('Choose one of "full", "knear", "k", "eps","e".'); 
end
end
