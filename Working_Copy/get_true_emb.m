function Em_true = get_true_emb(E,L)
k = max(L);
maxVal = 0;
for p = perms(1:k)
    X = -E;
    for i = 1:k
        X(X==-i) = p(i);
    end
    if sum(X==L) > maxVal
        maxVal = sum(X==L);
        Em_true = X;
    end
end
end
