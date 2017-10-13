function G = make_distance_graph_from_embeddings(Emb,varargin)
    n = size(Emb,1);
    G = zeros(n);
    for i = 1:n
        for j = 1:i-1
            G(i,j) = norm(Emb(i,:)-Emb(j,:));
        end
    end
    G = G + G';
    if nargin > 1
        dlmwrite(varargin{1},G,' ');
    end
end