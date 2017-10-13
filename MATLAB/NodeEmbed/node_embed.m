function [Em, ccr, nmi] = node_embed(G, labels, doNBT)
% Implements the node embeddings 'vec' algorithm of Ding et al. including a
% non-backtracking random walks option, different varieties of SGD (not yet
% fully implemented). Plots work with 2D data.
%
% Currently this version only works for matrices entirely in memory.
% Can input N, an nxd vector of data points, to create the graph using the
% given vectors in R^d, or leave it blank, which will run SBM.
%
% Brian Rappaport, 7/1/2017
% Modified 7/11/2017, Brian Rappaport - added aliasing method
% Modified 7/13/2017, Anu Gamage - added tsne visualization, metrics
% Modified 7/19/2017, Anu Gamage  - parallelized code to run on CPUs/GPUs
% Modified 7/24/2017, Brian Rappaport - standaradized codebase to combine
%                                       CPU and GPU usage

n = numel(labels);
k = max(labels);
P = create_aliases(G);
if gpuDeviceCount > 0
    useGPU = 1;
else
    useGPU = 0;
end
[Em, ccr, nmi] = run_node_embedding(P, labels, n, k, doNBT, useGPU);
end

function [Em_true, ccr, nmi] = run_node_embedding(P, labels, n, k, doNBT, useGPU)
% Runs the node embedding algorithm using the given parameters.

% define variables
rw_reps = 10; % number of random walks per data point
length = 60; % length of random walk
dim = 50; % dimension vectors are embedded into
win_size = 8; % size of window
neg_samples = 5; % number of negative samples
mb_size = 50; % size of mini-batches
max_reps = 2000; % maximum number of SGD iterations
do_plot = 0; % if want to plot intermediate results

% create positive samples
NN = cell(1,n);
parfor i = 1:n
    R = cell(1,rw_reps);
    for j = 1:rw_reps
        W = cell(1,win_size);
        rw = random_walk(i,length,P,doNBT);
        for w = 1:win_size
            W{w} = sparse(rw(1:end-w),rw(w+1:end),ones(numel(rw)-w,1),n,n,length-w);
        end
        R{j} = combine_cells(W,win_size);
    end
    NN{i} = combine_cells(R,rw_reps);
end
D_plus = combine_cells(NN,n);
% [i,j] = find(D_plus);
% D_plus_cell = accumarray(j,i,[n 1],@(v) {v.'});

% create negative samples
MM = cell(1,n);
parfor i = 1:n
    num = neg_samples;
    MM{i} = sparse(linspace(i,i,num),randi(n,1,num),linspace(1,1,num),n,n,num);
end
D_minus = combine_cells(MM,n);

% begin SGD
if useGPU
    U = rand(n,dim,'gpuArray');
else
    U = rand(n,dim);
end
[npx,npy] = find(D_plus);
[nmx,nmy] = find(D_minus);
nnzp = numel(npx);
nnzm = numel(nmx);
gradTermP = @(u,v,U) bsxfun(@rdivide,-U(v,:),diag(1+exp(U(u,:)*U(v,:)')));
gradTermM = @(u,v,U) bsxfun(@rdivide,U(v,:),diag(1+exp(-U(u,:)*U(v,:)')));
for rep = 1:max_reps
    mu = sqrt(log(max_reps)/rep);
    i = randi(nnzp,mb_size,1);
    j = randi(nnzm,mb_size,1);
    if useGPU
        [px,pinds] = unique(gpuArray(npx(i)));
        py = gpuArray(npy(i));
        py = py(pinds);
        [mx,minds] = unique(gpuArray(nmx(j)));
        my = gpuArray(nmy(j));
        my = my(minds);
    else
        px = npx(i);
        py = npy(i);
        pinds = 1:mb_size;
        mx = nmx(j);
        my = nmy(j);
        minds = 1:mb_size;
    end
    gradP = gradTermP(px,py,U);
    gradM = gradTermM(mx,my,U);
    U(px,:) = U(px,:) - mu*gradP(pinds,:);
    U(mx,:) = U(mx,:) - mu*gradM(minds,:);
end
% run k-means
Em = kmeans(U,k);

% relabel nodes
Em_true = get_true_emb(Em,labels);

ccr = sum(Em_true == labels')*100/n;
nmi = get_nmi(labels', Em_true);

% plot results
if do_plot
    plot_results(N,G,Em_true);
    
    % confusion matrix
    targets = zeros(k,n);
    for i=1:k
        targets(i,(labels == i)) = 1;
    end
    outputs = zeros(k,n);
    for i=1:n
        outputs(Em_true(i),i) = 1;
    end
    figure(102);
    plotconfusion(targets, outputs);
end
end

function C = combine_cells(R,i)
% A helper function to reduce the time and space required for making the 
% random walks matrix.
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
% represented as 1i for lack of a better way of dealing with it.
d = 0;
parfor i = 1:numel(nzP_X)
    e = log(1+exp(-U(nzP_X(i),:)*U(nzP_Y(i),:)'));
    if e==inf, e=1i; end
    d = d + e;
end
parfor i = 1:numel(nzM_X)
    e = log(1+exp(U(nzM_X(i),:)*U(nzM_Y(i),:)'));
    if e==inf, e=1i; end
    d = d + e;
end
end

function plot_results(N, G, Em)
% Plots the results of the algorithm. This may be outdated.
figure(101);
   hold on;
   if size(N,1) == 2  % Plot results for 2D data
       for i = 1:k
           gplot(G(Em==i,Em==i),N(Em==i,:),'-o');
       end
       title(sprintf('%d data points in %d clusters',n,k));
   elseif size(N,1) > 2 % Use tsne to visualize higher dimensional results
       mappedX = tsne(U, labels, 2, dim, 30);
        for i = 1:k
            scatter(mappedX(Em==i,1),mappedX(Em==i,2),'o');
        end
        title(sprintf('%d data points in %d clusters, visualized using TSNE',n,k));
   end
   
end
