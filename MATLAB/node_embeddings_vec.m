function [Em, ccr_val, nmi_val] = node_embeddings_vec(varargin)                                   
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

rw_reps = 10; % number of random walks per data point                       
length = 60; % length of random walk                                        
dim = 50; % dimension vectors are embedded into
win_size = 8; % size of window
neg_samples = 5; % number of negative samples
mb_size = 50; % size of mini-batches
gamma = .2; % momentum term
max_reps = 1000; % maximum number of SGD iterations
do_plot = 0; % if want to plot intermediate results

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
    do_non_backtracking = varargin{5};
    [G, labels] = make_SBM(n,k,'const',c,lambda); % alternate graph making method    
    [v,~] = eigs(diag(sum(G))-G,3);
    N = v(:,[2 3]);
end

% create random walk matrix
P = create_aliases(G);
D_plus = sparse([],[],false,n,n,ceil(n*n*.1));
for i = 1:n
    D_plus_mid = sparse([],[],false,n,n,n);
    for j = 1:rw_reps
        D_plus_inner = sparse([],[],false,n,n,n);
        rw = random_walk(i,length,P,do_non_backtracking);
        for w = 1:win_size
            D_plus_inner = D_plus_inner + sparse(rw(1:end-w),rw(w+1:end),ones(numel(rw)-w,1),n,n,length-w);
        end
        D_plus_mid = D_plus_mid + D_plus_inner;
    end
    D_plus = D_plus + D_plus_mid;
end

% create negative samples
D_minus = sparse([],[],false,n,n,ceil(neg_samples*n*n*.1));
num_elems = sum(D_plus,1);
for i = 1:n
    num = neg_samples * num_elems(i);
    D_minus = D_minus + sparse(linspace(i,i,num),randi(n,1,num),linspace(1,1,num),n,n,num);
end

% begin SGD
U = rand(n,dim);
[nzP_X,nzP_Y] = find(D_plus);
[nzM_X,nzM_Y] = find(D_minus);
nnzP = numel(nzP_X);
nnzM = numel(nzM_X);
gradTermP = @(i1,i2,U) bsxfun(@rdivide,-U(i2,:),diag(1+exp(U(i1,:)*U(i2,:)')));
gradTermM = @(i1,i2,U) bsxfun(@rdivide,U(i2,:),diag(1+exp(-U(i1,:)*U(i2,:)')));
cost = zeros(1,max_reps);
gradP = 0;
gradM = 0;
for rep = 1:max_reps
    mu = sqrt(log(max_reps)/(2*rep));
    i = randi(nnzP,mb_size,1);
    gradP = gamma*gradP + mu*gradTermP(nzP_X(i),nzP_Y(i),U);
    j = randi(nnzM,mb_size,1);
    gradM = gamma*gradM + mu*gradTermM(nzM_X(j),nzM_Y(j),U);
    U(nzP_X(i),:) = U(nzP_X(i),:) - gradP;
    U(nzM_X(j),:) = U(nzM_X(j),:) - gradM; 
    if rep > 10
        if 1-real(cost(rep-10)/cost(rep)) < .015 && real(cost(rep-1)/cost(rep)) <= 1
            patience = patience + 1;
            if patience == 3
                break
            end
        else
            patience = 0;
        end
    end 
    if do_plot && rem(rep,10) == 0
        cost(rep) = cost_fn(U,nzP_X,nzP_Y,nzM_X,nzM_Y);  
        disp([num2str(rep) ' cost is ' num2str(cost(rep))]);
        if size(N,2) == 2
            Em = kmeans(U,k);
            figure(100);
            clf
            hold on;
            axis equal;
            for i = 1:k
                gplot(G(Em==i,Em==i),N(Em==i,:),'-o');
            end
            pause(.02);
        end
    end
end
Em = kmeans(U,k);
if do_plot && size(N,1) == 2  % Plot results for 2D data
    figure;
    hold on;
    for i = 1:k
        gplot(G(Em==i,Em==i),N(Em==i,:),'-o');
    end
    title(sprintf('%d data points, %d clusters',n,k));
    figure;
    plot(cost(10:10:end));
    title('Cost over time');
end

% Use tsne to visualize  higher dimensional results
if do_plot && dim > 2
    mappedX = tsne(U, labels, 2, dim, 30);
    % Plot results
    for i=1:k
        cluster = mappedX(Em == i, :);
        scatter(cluster(:,1), cluster(:,2), 'o')
        hold on
        pause(1)
    end
    title('Clusters Visualized using TSNE')
end

% Relabelling with the correct labels
true_label = zeros(k,1);
for i=1:k
    %fprintf('Label_pred: %d\n', i);
    x = labels(Em==i);
    freq = zeros(1, k);
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
    
    %fprintf('Label_true:%d\n\n', true_label(i))
end

Em_true = zeros(1,n);
for i = 1:k
    Em_true(Em==i) = true_label(i);
end

% Confusion matrix
targets = zeros(k,n);
for i=1:k
   targets(i, (labels == i)) = 1; 
end

outputs = zeros(k,n);
for i=1:n
    outputs(Em_true(i),i) = 1; 
end

if do_plot
    figure
    plotconfusion(targets, outputs)
end

ccr_val = (1 - confusion(targets, outputs))*100;
nmi_val = nmi(labels, Em_true);

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

function P = create_aliases(G)
% Initializes aliasing array for fast choice from discrete random
% distribution
n = size(G,1);
P = cell(1,n);
for i = 1:n
    [~,Ginds,I] = find(G(i,:));
    k = numel(I);
    if k == 0, continue; end
    list = zeros(k,3);
    [I,inds] = sort(I/sum(I)*k);
    loc = find(I>1,1);
    if isempty(loc), loc = 1; end
    list(1:loc-1,1) = Ginds(inds(1:loc-1));
    list(1:loc-1,3) = I(1:loc-1);
    ind = 1;
    for f = loc:k
        while I(f) > 1
            list(ind,2) = Ginds(inds(f));
            I(f) = I(f) - (1 - list(ind,3));
            ind = ind + 1;
        end
        list(f,1) = Ginds(inds(f));
        list(f,3) = I(f);
    end
    P{i} = list;
end
end

function s = random_walk(curr,length,P,doNBT)
% Runs a random walk on positive values of G of length 'length' starting at
% 'curr'. If doNBT is set to 1, the function will ensure the random walk
% doesn't backtrack either. P contains the aliasing array to make
% generation of next step faster
s = zeros(1,length);
prev = -1;
for i = 1:length
    s(i) = curr;
    alias = P{i};
    if isempty(alias)
        s(i+1:end) = [];
        break
    end
    if doNBT
        x = find(prev==alias(:,1),1);
        if ~isempty(x)
            alias = alias([1:x-1,x+1:end],:);
            if isempty(alias)
                s(i+1:end) = [];
                break
            end
        end
        prev = curr;
    end
    x = ceil(rand()*size(alias,1));
    prob = alias(x,3);
    if rand() > prob
        curr = alias(x,2);
    else
        curr = alias(x,1);        
    end
end
end

function G = make_graph(N,type,varargin)                              
% Creates the similarity graph. 
% Copyright (c) 2012, Ingo Bork
% Copyright (c) 2003, Jochen Lenz
% Copyright (c) 2013, Oliver Woodford
% Edits 2017, Brian Rappaport
n = size(N,1);
N = N';
G = zeros(n,n);
dist = @(x,y) exp(-norm(x-y));
switch type
    case 'full'
        for i = 1:n
            for j = 1:i-1
                G(i,j) = dist(N(i,:),N(j,:));
            end
        end
        G = G + G';
    case {'knear','k'}
        k = varargin{1};
        for i = 1:n
            dist = sqrt(sum((repmat(N(:,i),1,n) - N) .^ 2, 1));
            [s,O] = sort(dist,'ascend');
            indi(1,(i-1)*k+1:i*k) = i;
            indj(1,(i-1)*k+1:i*k) = O(1:k);
            inds(1,(i-1)*k+1:i*k) = s(1:k);
        end
        G = sparse(indi,indj,inds,n,n);
        G = max(G,G');
    case 'eps'
        e = varargin{1};
        indi = [];
        indj = [];
        for i = 1:n
            dist = sqrt(sum((repmat(N(:,i),1,n) - N) .^ 2, 1));
            dist = (dist < e);
            last = size(indi,2);
            count = nnz(dist);
            [~,col] = find(dist);
            indi(1,last+1:last+count) = i;
            indj(1,last+1:last+count) = col;
        end
        G = sparse(indi,indj,ones(1,numel(indi)),n,n);
    otherwise
        error('Choose one of "full", "knear", "k", "eps".'); 
end
end

function [G, labels] = make_SBM(n,k,scaling_type,c,lambda)                  % modified to return labels
% n is the number of nodes
% k is the number of communities
% scaling_type is either constant or logarithmic
% c is the constant fraction associated with the scaling
% lambda is the reduction percentage
% Brian Rappaport, 7/6/17

% We're using equal partitions
P = repmat(1:k,ceil(n/k),1);
P = P(1:n);
labels = P';
% get connections
if strcmp(scaling_type,'const')
    % odds if in same community is c, else is c(1-lambda)
    q = c*(1-lambda);
elseif strcmp(scaling_type,'log')
    % odds if in same community is clog(n), else is c(1-lambda)log(n)
    q = c*(1-lambda)*log(n);
else
    error('scaling type must be ''const'' or ''log''');
end
% build graph
indI = [];
indJ = [];
for i = 1:n
    I = rand(1,n);
    ind = P(i)==P;
    I(ind) = I(ind)*(1-lambda);
    f = find(I<q);
    indI(end+1:end+numel(f)) = f;
    indJ(end+1:end+numel(f)) = i;
end
G = sparse(indI,indJ,ones(numel(indI),1),n,n,numel(indI));
G = max(G,G');
end