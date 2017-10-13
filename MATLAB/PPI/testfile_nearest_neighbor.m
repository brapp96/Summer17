num_trials = 10;
remove = 200;
neighbors = 300;

load('runs/PPIlong.mat');
VECpath = 'MIPS_data/VEC_distance_graph.txt';
if exist('VEC','var')
elseif exist(VECpath,'file')
    VEC = dlmread(VECpath);
else
    VEC = make_distance_graph_from_embeddings(NBT{5},VECpath);
end
if ~exist('DSD','var') || ~exist('protein_names','var')
    [DSD,protein_names] = read_DSD_file('MIPS_data/results_converged.DSD1');
end
if ~exist('annos','var') || ~exist('annos_names','var')
    [annos_names,annos] = read_first_level('MIPS_data/MIPSFirstLevel.list');
end

% Force annos_names and protein_names to match by removing those not in both
% Assuming protein_names is a superset of annos_names
[~,bad_inds] = setdiff(protein_names,annos_names);
good_inds = setdiff(1:numel(protein_names),bad_inds);
[~,protein_inds] = sort(protein_names(good_inds));
DSD_filt = DSD(protein_inds,protein_inds);
VEC_filt = VEC(protein_inds,protein_inds);
annos_filt = annos(protein_inds);

dsd_results = zeros(num_trials,2);
vec_results = zeros(num_trials,2);
for ii = 1:num_trials
    for jj = 0:1
        dsd_results(ii,jj+1) = nearest_neighbor(DSD_filt,annos_filt,protein_inds,remove,neighbors,jj);
        vec_results(ii,jj+1) = nearest_neighbor(VEC_filt,annos_filt,protein_inds,remove,neighbors,jj);
    end
end

figure(1);
clf
plot(dsd_results(:,1));hold on;plot(vec_results(:,1));
title('Unweighted');
axis([0 num_trials 0 1]);
legend('DSD','VEC');
saveas(gcf,['figs/Unweighted_T' num2str(neighbors) '_num2rem' num2str(remove) '.png']);
figure(2);
clf
plot(dsd_results(:,2));hold on;plot(vec_results(:,2));
title('Weighted');
axis([0 num_trials 0 1]);
legend('DSD','VEC');
saveas(gcf,['figs/Weighted_T' num2str(neighbors) '_num2rem' num2str(remove) '.png']);