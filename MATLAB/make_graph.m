
function G = make_graph(N,type,varargin)                              
% Creates the similarity graph. 
% Copyright (c) 2012, Ingo Bork
% Copyright (c) 2003, Jochen Lenz
% Copyright (c) 2013, Oliver Woodford
% Edits 2017, Brian Rappaport
n = size(N,1);
N = N';
G = zeros(n,n);
dist = @(x,y) exp(-norm(x-y));
switch type
    case 'full'
        for i = 1:n
            for j = 1:i-1
                G(i,j) = dist(N(i,:),N(j,:));
            end
        end
        G = G + G';
    case {'knear','k'}
        k = varargin{1};
        for i = 1:n
            dist = sqrt(sum((repmat(N(:,i),1,n) - N) .^ 2, 1));
            [s,O] = sort(dist,'ascend');
            indi(1,(i-1)*k+1:i*k) = i;
            indj(1,(i-1)*k+1:i*k) = O(1:k);
            inds(1,(i-1)*k+1:i*k) = s(1:k);
        end
        G = sparse(indi,indj,inds,n,n);
        G = max(G,G');
    case 'eps'
        e = varargin{1};
        indi = [];
        indj = [];
        for i = 1:n
            dist = sqrt(sum((repmat(N(:,i),1,n) - N) .^ 2, 1));
            dist = (dist < e);
            last = size(indi,2);
            count = nnz(dist);
            [~,col] = find(dist);
            indi(1,last+1:last+count) = i;
            indj(1,last+1:last+count) = col;
        end
        G = sparse(indi,indj,ones(1,numel(indi)),n,n);
    otherwise
        error('Choose one of "full", "knear", "k", "eps".'); 
end
end
