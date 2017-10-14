function [accuracy,f1] = nearest_neighbor(graph,annos,protein_inds,num2remove,t,is_weighted)
    alpha = 3;
    n = numel(protein_inds);
    numCorrect = 0;
    f1_scores = zeros(3,43); % 1st row: true positives; 2nd row: false positives; 3rd row: false negatives
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
        true_pos = intersect(max_GOs,orig_GOs);
        false_pos = setdiff(max_GOs,orig_GOs);
        false_neg = setdiff(orig_GOs,max_GOs);
        f1_scores(1,true_pos) = f1_scores(1,true_pos) + 1;
        f1_scores(2,false_pos) = f1_scores(2,false_pos) + 1;
        f1_scores(3,false_neg) = f1_scores(3,false_neg) + 1;
    end
    accuracy = numCorrect/num2remove;
    f1 = inf(1,43);
    for ii = 1:43
        if sum(f1_scores(:,ii)) == 0
            continue
        end
        f1(ii) = 2*f1_scores(1,ii)/(2*f1_scores(1,ii) + f1_scores(2,ii) + f1_scores(3,ii));
    end
    f1 = sum(f1(f1~=inf))/sum(f1~=inf);
end
