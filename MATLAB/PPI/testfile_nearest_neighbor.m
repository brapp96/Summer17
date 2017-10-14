num_trials = 10;
remove = 100;
neighbors = 500;
dataPrefix = 'MIPS_data/';
edgesFile = [dataPrefix 'physical.PPI'];
adjListFile = [dataPrefix 'PPIadj.txt'];
DSDFile = [dataPrefix 'DSD_graph.txt'];
AnnosFile = [dataPrefix 'MIPSFirstLevel.list'];

tic;
system(sprintf('python DSD_python_code/DSDmain.py -c -q -o %s %s',DSDFile, edgesFile));
[DSD,protein_names] = read_DSD_file(DSDFile);
tDSD = toc;
fprintf('time taken for DSD embedding: %f seconds',tDSD);

tic;
len = 60;
VEC_graph = import_graph_by_edges(adjListFile);
NBT_emb = node_embed_file(VEC_graph,0,1,len);
VEC = make_distance_graph_from_embeddings(NBT_emb);
tVEC = toc;
fprintf('time taken for VEC embedding: %f seconds',tVEC);

[annos_names,annos] = read_first_level(AnnosFile);

% load('runs/PPIlong.mat');
% VECpath = 'MIPS_data/VEC_distance_graph.txt';
% if exist('VEC','var')
% elseif exist(VECpath,'file')
%     VEC = dlmread(VECpath);
% else
%     VEC = make_distance_graph_from_embeddings(NBT{5},VECpath);
% end
% if ~exist('DSD','var') || ~exist('protein_names','var')
%     [DSD,protein_names] = read_DSD_file('MIPS_data/DSD_graph.txt');
% end
% if ~exist('annos','var') || ~exist('annos_names','var')
%     [annos_names,annos] = read_first_level('MIPS_data/MIPSFirstLevel.list');
% end

% Force annos_names and protein_names to match by removing those not in both
% Assuming protein_names is a superset of annos_names
[~,bad_inds] = setdiff(protein_names,annos_names);
good_inds = setdiff(1:numel(protein_names),bad_inds);
[~,protein_inds] = sort(protein_names(good_inds));
DSD_filt = DSD(protein_inds,protein_inds);
VEC_filt = VEC(protein_inds,protein_inds);
annos_filt = annos(protein_inds);

dsd_acc = zeros(num_trials,2);
dsd_f1 = zeros(num_trials,2);
vec_acc = zeros(num_trials,2);
vec_f1 = zeros(num_trials,2);
for ii = 1:num_trials
    for jj = 0:1
        [dsd_acc(ii,jj+1),dsd_f1(ii,jj+1)] = nearest_neighbor(DSD_filt,annos_filt,protein_inds,remove,neighbors,jj);
        [vec_acc(ii,jj+1),vec_f1(ii,jj+1)] = nearest_neighbor(VEC_filt,annos_filt,protein_inds,remove,neighbors,jj);
    end
end

figure(1);
clf
plot(dsd_acc(:,1));hold on;plot(vec_acc(:,1));
title('Unweighted Accuracy');
axis([0 num_trials 0 1]);
legend('DSD','VEC');
saveas(gcf,['figs/Acc_unweighted_T' num2str(neighbors) '_num2rem' num2str(remove) '.png']);
figure(2);
clf
plot(dsd_acc(:,2));hold on;plot(vec_acc(:,2));
title('Weighted Accuracy');
axis([0 num_trials 0 1]);
legend('DSD','VEC');
saveas(gcf,['figs/Acc_weighted_T' num2str(neighbors) '_num2rem' num2str(remove) '.png']);

figure(3);
clf
plot(dsd_f1(:,1));hold on;plot(vec_f1(:,1));
title('Unweighted F1 Score');
axis([0 num_trials 0 1]);
legend('DSD','VEC');
saveas(gcf,['figs/F1_unweighted_T' num2str(neighbors) '_num2rem' num2str(remove) '.png']);
figure(4);
clf
plot(dsd_f1(:,2));hold on;plot(vec_f1(:,2));
title('Weighted F1 Score');
axis([0 num_trials 0 1]);
legend('DSD','VEC');
saveas(gcf,['figs/F1_weighted_T' num2str(neighbors) '_num2rem' num2str(remove) '.png']);