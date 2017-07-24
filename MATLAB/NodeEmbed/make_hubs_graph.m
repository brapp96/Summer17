function [G,P] = make_hubs_graph(n, K, H, p, ph, pd)
% n is number of points
% K is number of communities
% H is number of hubs
% p is probability node connects at all
% ph is probability node connects to a hub
% pd is probability node connects to different comm

% Build graph
hubArr = [ph 1-ph];
indI = [];
indJ = [];
k = 0;
hub = 1;
for i = 1:n
    k = k+1;
    if k == K+1
        k = 1;
        H = H - 1;
        if H < 1
            hub = 2; 
        end
    end
    I = rand(1,n)/hubArr(hub)/p;
    ind = k:K:n;
    I(ind) = I(ind)/pd;
    f = find(I<1);
    indI(end+1:end+numel(f)) = f;
    indJ(end+1:end+numel(f)) = i;
end
G = sparse(indI,indJ,ones(numel(indI),1),n,n,numel(indI));
G = max(G,G');
% G = [G(2:2:end,1:2:end) G(1:2:end,1:2:end);G(2:2:end,2:2:end) G(1:2:end,2:2:end)];
end