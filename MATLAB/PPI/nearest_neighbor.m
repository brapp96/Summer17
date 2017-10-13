function accuracy = nearest_neighbor(graph,annos,protein_inds,num2remove,t,is_weighted)
    alpha = 3;
    n = numel(protein_inds);
    numCorrect = 0;
    for ii = 1:num2remove
        selectedElement = randi(n,1);
        if is_weighted
            [mvals,minds] = sort(graph(selectedElement,:));
        else
            [~,minds] = sort(graph(selectedElement,:));
        end
        if ~is_weighted
            invmvals = ones(1,t+1);
        else
            invmvals = 1./mvals(1:t+1);
        end        
        votes = zeros(1,43);
        for jj = 2:t+1 % strip out first element - always 0
            thisVote = annos{minds(jj)};
            votes(thisVote) = votes(thisVote) + invmvals(jj);
        end
       % votes = hist(vertcat(annos{minds(2:t+1)}),1:43); a little too hacky
        [~,voteinds] = sort(votes);
        max_GOs = voteinds(end:-1:end-(alpha-1));
        orig_GOs = annos{selectedElement};
        if isempty(orig_GOs)
            continue;
        end
        accuracy_matrix = max_GOs==orig_GOs;
        if any(accuracy_matrix(:))
            numCorrect = numCorrect+1;
        end
    end
    accuracy = numCorrect/num2remove;
end
