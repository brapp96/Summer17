function G = make_distance_graph_from_embeddings(Emb,varargin)
    n = size(Emb,1);
    G = zeros(n);
    EmbNorm = Emb./sum(Emb.^2,2);
    for i = 1:n
        for j = 1:i-1
            G(i,j) = EmbNorm(i,:)*EmbNorm(j,:)';
        end
    end
    G = G/max(G(:));
    G = G + G';
    if nargin > 1
        dlmwrite(varargin{1},G,' ');
    end
end