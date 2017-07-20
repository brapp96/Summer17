function [Em_brw, ccr_brw, nmi_brw, Em_nbrw, ccr_nbrw, nmi_nbrw] = node_embed_gpu(varargin)                                   
% Implements the node embeddings 'vec' algorithm of Ding et al. including a
% non-backtracking random walks option, different varieties of SGD (not yet
% fully implemented). Plots work with 2D data.
%
% Currently this version only works for matrices entirely in memory.
% Can input N, an nxd vector of data points, to create the graph using the
% given vectors in R^d, or leave it blank, which will run SBM.
% 
% Creation 7/1/2017, Brian Rappaport
% Modified 7/11/2017, Brian Rappaport - added aliasing method
% Modified 7/13/2017, Anu Gamage - added tsne visualization, metrics
% Modified 7/19/2017, Anu Gamage  - parallelized code to run on CPUs/GPUs

% create graph
if nargin == 1
    N = varargin{1};
    n = size(N,1);
    k = 2; % default number of clusters  
    G = make_graph(N,'k',20); % 'full'; 'k' or 'knear' with k; 'eps' with eps
else
    n = varargin{1};
    k = varargin{2};
    c = varargin{3};
    lambda = varargin{4};
    [G, labels] = make_SBM(n,k,'const',c,lambda); % alternate graph making method    
%    [v,~] = eigs(diag(sum(G))-G,3);
%    N = v(:,[2 3]);
end


% create random walk matrix using BRW/NBRW
P = create_aliases(G);


[Em_brw, ccr_brw, nmi_brw] = run_node_embedding(P, labels, n, k, 0);
[Em_nbrw, ccr_nbrw, nmi_nbrw] = run_node_embedding(P, labels, n, k, 1);


end

function [Em_true, ccr_val, nmi_val] = run_node_embedding(P, labels, n, k,do_non_backtracking)

rw_reps = 10; % number of random walks per data point                       
length = 60; % length of random walk                                        
dim = 50; % dimension vectors are embedded into
win_size = 8; % size of window
neg_samples = 5; % number of negative samples
mb_size = 50; % size of mini-batches
gamma = .2; % momentum term
max_reps = 1000; % maximum number of SGD iterations
do_plot = 0; % if want to plot intermediate results


    NN = cell(1,n);
    parfor i = 1:n
        R = cell(1,rw_reps);
        for j = 1:rw_reps
            W = cell(1,win_size);
            rw = random_walk(i,length,P,do_non_backtracking);
            for w = 1:win_size
                W{w} = sparse(rw(1:end-w),rw(w+1:end),ones(numel(rw)-w,1),n,n);
            end
            R{j} = combine_cells(W,win_size);
        end
        NN{i} = combine_cells(R,rw_reps);
    end
    D_plus = combine_cells(NN,n);
    
    % create negative samples
    MM = cell(1,n);
    %num_elems = full(sum(D_plus,1));
    parfor i = 1:n
       %num = neg_samples * num_elems(i);
       num = neg_samples;
       MM{i} = sparse(linspace(i,i,num),randi(n,1,num),linspace(1,1,num),n,n,num);
    end
%     ival = repmat(gpuArray.linspace(1, n, n)',neg_samples, 1);
%     jval = randi(n, [n*neg_samples,1], 'gpuArray');
%     val =  repmat(gpuArray.linspace(1, 1, n)',neg_samples, 1);
%     D_minus = spconvert([ival, jval, val]);
    %MM = arrayfun(@sparse, ival, jval, val);
    D_minus = combine_cells(MM,n);
    
    
    % begin SGD
    U = rand(n,dim, 'gpuArray');
    [nzP_X,nzP_Y] = find(D_plus);
    [nzM_X,nzM_Y] = find(D_minus);
    nnzP = numel(nzP_X);
    nnzM = numel(nzM_X);
    gradTermP = @(i1,i2,U) bsxfun(@rdivide,-U(i2,:),diag(1+exp(U(i1,:)*U(i2,:)')));
    gradTermM = @(i1,i2,U) bsxfun(@rdivide,U(i2,:),diag(1+exp(-U(i1,:)*U(i2,:)')));
    % cost = zeros(1,max_reps);
    gradP = 0;
    gradM = 0;
    for rep = 1:max_reps
       % disp(rep)
        mu = sqrt(log(max_reps)/(2*rep));
        i = gpuArray.randperm(nnzP);
        [~, ui] = unique(nzP_X(i));     %> nzP_X(i(ui(1)))
        i = i(ui);
        i = i(1:mb_size);
        
        gradP = gamma*gradP + mu*gradTermP(nzP_X(i),nzP_Y(i),U);
        
        j = gpuArray.randperm(nnzM);
        [~, uj] = unique(nzM_X(j));
        j = j(uj);
        j = j(1:mb_size);
        gradM = gamma*gradM + mu*gradTermM(nzM_X(j),nzM_Y(j),U);
        
      %  for z = 1:mb_size
      %      U(nzP_X(i(z)),:) = U(nzP_X(i(z)),:) - gradP(z);
      %      U(nzM_X(j(z)),:) = U(nzM_X(j(z)),:) - gradM(z); 
      %  end
      
        U(nzP_X(i),:) = U(nzP_X(i),:) - gradP;
        U(nzM_X(j),:) = U(nzM_X(j),:) - gradM; 
%        if rem(rep,10) == 0
%            disp(rep)
%        end

    %     if rep > 10
    %         if 1-real(cost(rep-10)/cost(rep)) < .015 && real(cost(rep-1)/cost(rep)) <= 1
    %             patience = patience + 1;
    %             if patience == 3
    %                 break
    %             end
    %         else
    %             patience = 0;
    %         end
    %     end 
    %     if do_plot && rem(rep,10) == 0
    %         cost(rep) = cost_fn(U,nzP_X,nzP_Y,nzM_X,nzM_Y);  
    %         disp([num2str(rep) ' cost is ' num2str(cost(rep))]);
    %         if size(N,2) == 2
    %             Em = kmeans(U,k);
    %             figure(100);
    %             clf
    %             hold on;
    %             axis equal;
    %             for i = 1:k
    %                 gplot(G(Em==i,Em==i),N(Em==i,:),'-o');
    %             end
    %             pause(.02);
    %         end
    %     end
    end
    Em = kmeans(U,k);
    
    % Plot results
    %if do_plot && 0
    %    figure(101);
    %    hold on;
    %    if size(N,1) == 2  % Plot results for 2D data
    %        for i = 1:k
    %            gplot(G(Em==i,Em==i),N(Em==i,:),'-o');
    %        end
    %        title(sprintf('%d data points in %d clusters',n,k));
    %    elseif size(N,1) > 2 % Use tsne to visualize higher dimensional results
    %        mappedX = tsne(U, labels, 2, dim, 30);
    %         for i = 1:k
    %             scatter(mappedX(Em==i,1),mappedX(Em==i,2),'o');
    %         end
    %         title(sprintf('%d data points in %d clusters, visualized using TSNE',n,k));
    %    end
    %end
    
    % figure
    % scatter(mappedX(Em==1,1),mappedX(Em==1,2),'bo');
    % hold on
    % scatter(mappedX(Em==2,1),mappedX(Em==2,2),'ro');
    
    
    % Relabelling with the correct labels
    true_label = gpuArray.zeros(k,1);
    for i = 1:k
        x = labels(Em==i);
        freq = gpuArray.zeros(1, k);
        for p = 1:k
            freq(p) = sum(x==p);
        end
        for j=1:k
            [~, max_c] = max(freq);
            if(~ismember(max_c, true_label))
                true_label(i) = max_c;
                break;
            else 
                freq(max_c) = 0;            
            end
        end
    end
    Em_true = gpuArray.zeros(1,n);
    for i = 1:k
        Em_true(Em==i) = true_label(i);
    end
    
    % Confusion matrix
    targets = gpuArray.zeros(k,n);
    for i=1:k
       targets(i, (labels == i)') = 1; 
    end
    
    outputs = gpuArray.zeros(k,n);
    for i=1:n
        outputs(Em_true(i),i) = 1; 
    end
    
    if do_plot 
        figure(101);
        plotconfusion(targets, outputs)
    end
    
    
    ccr_val = (sum(Em_true == labels')/n)*100;
    nmi_val = nmi(labels', Em_true);
       
end

function C = combine_cells(R,i)
    while i ~= 1
        if rem(i,2) == 1
            R{i-1} = R{i-1} + R{i};
            i = i-1;
        end
        for ii = 1:i/2
            R{ii} = R{ii} + R{ii+i/2};
        end
        i = i/2;
    end
    C = R{1};
end

function d = cost_fn(U,nzP_X,nzP_Y,nzM_X,nzM_Y)
    % Calculates cost function. If any given cost overflows it is
    % represented as 1i for lack of a better way of dealing with it
    d = 0;
    for i = 1:numel(nzP_X)
        e = log(1+exp(-U(nzP_X(i),:)*U(nzP_Y(i),:)'));
        if e==inf, e=1i; end
        d = d + e;
    end
    for i = 1:numel(nzM_X)
        e = log(1+exp(U(nzM_X(i),:)*U(nzM_Y(i),:)'));
        if e==inf, e=1i; end
        d = d + e;
    end
end
