function accuracy = nearest_neighbor(graph,protein_names,annos,annos_names,num2remove,t,is_weighted)
    alpha = 3;
    num_annos = numel(annos_names);
    test_set = cell(1,num2remove);
    i = num2remove;
    numCorrect = 0;
    while i > 0
        name = annos_names(randi(num_annos,1));
        if any(strcmp(name,protein_names))
            test_set(i) = name;
            i = i-1;
        end
    end
    for i = 1:num2remove
        [mvals,minds] = sort(graph(strcmp(protein_names,test_set(i)),:));
        t_nearest = protein_names(minds(2:t+1)); % strip out first element - always 0
        vals_nearest = mvals(2:t+1);
        if ~is_weighted
            vals_nearest = ones(1,t);
        end
        votes = zeros(1,43);
        for j = 1:t
            temp = strcmp(annos_names,t_nearest(j));
            if ~any(temp)
                continue;
            end
            thisVote = annos{temp};
            votes(thisVote) = votes(thisVote) + 1/vals_nearest(j);
        end
        [~,voteinds] = sort(votes);
        max_GOs = voteinds(end:-1:end-(alpha-1));
        orig_GOs = annos{strcmp(annos_names,test_set(i))};
        M1 = repmat(max_GOs,numel(orig_GOs),1);
        M2 = repmat(orig_GOs,1,alpha);
        if isempty(M2)
            continue;
        end
        accuracy_matrix = M1==M2;
        if any(accuracy_matrix(:))
            numCorrect = numCorrect+1;
        end
    end
    accuracy = numCorrect/num2remove;
end
