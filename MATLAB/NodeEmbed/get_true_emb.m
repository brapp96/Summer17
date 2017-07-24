function Em_true = get_true_emb(Em,labels)
n = numel(labels);
k = max(labels);
true_label = zeros(k,1);
for i = 1:k
    x = labels(Em==i);
    freq = zeros(1, k);
    for p = 1:k
        freq(p) = sum(x==p);
    end
    for j = 1:k
        [~, max_c] = max(freq);
        if(~ismember(max_c, true_label))
            true_label(i) = max_c;
            break;
        else
            freq(max_c) = 0;
        end
    end
end
Em_true = zeros(1,n);
for i = 1:k
    Em_true(Em==i) = true_label(i);
end
end
