function [G,P] = make_hubs_graph(n, k, H, p, ph, pd)
% n is number of points
% K is number of communities
% H is number of hubs
% p is probability node connects at all
% ph is probability node connects to a hub
% pd is probability node connects to different community
%
% works like this: an edge from between a hub and another node in a community 
% has probability p of forming. If the edge is between two non-hubs, it has
% probability p*ph of forming. If it is between a hub and another node in a
% separate community, it has probability p*pd of forming. If it's between
% two nonhubs in different communities, it has probability p*ph*pd of
% forming.
%
% Brian Rappaport, 7/24/17

indI = [];
indJ = [];
kk = 0;
hub = 1;
P = ones(1,n);
for i = 1:n
    kk = kk+1;
    if kk == k+1
        kk = 1;
        H = H - 1;
        if H < 1
            hub = 0;
        end
    end
    P(i) = kk;
    I = rand(1,n);
    if ~hub
        I = I/(1-ph);
    end
    others = kk:k:n;
    I(others) = I(others)/pd;
    f = find(I<p);
    indI(end+1:end+numel(f)) = f;
    indJ(end+1:end+numel(f)) = i;
end
G = sparse(indI,indJ,ones(numel(indI),1),n,n,numel(indI));
G = triu(G)+triu(G)';
% G = [G(2:2:end,1:2:end) G(1:2:end,1:2:end);G(2:2:end,2:2:end) G(1:2:end,2:2:end)];
end