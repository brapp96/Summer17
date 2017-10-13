num_trials = 10;
remove = 50;
neighbors = 10;
weighted = 1;
load('runs/PPI.mat');
VEC = NBT{6};

protein_names = get_protein_names('MIPS_data/proteinNames.txt');
[annos,annos_names] = read_first_level('MIPS_data/MIPSFirstLevel.list');

dsd_results_unweighted = zeros(1,num_trials);
dsd_results_weighted = zeros(1,num_trials);
vec_results_unweighted = zeros(1,num_trials);
vec_results_weighted = zeros(1,num_trials);
for ii = 1:num_trials
    dsd_results_unweighted(ii) = nearest_neighbor(DSD,protein_names,annos,annos_names,remove,neighbors,~weighted);
    dsd_results_weighted(ii) = nearest_neighbor(DSD,protein_names,annos,annos_names,remove,neighbors,weighted);
    vec_results_unweighted(ii) = nearest_neighbor(VEC,protein_names,annos,annos_names,remove,neighbors,~weighted);
    vec_results_weighted(ii) = nearest_neighbor(VEC,protein_names,annos,annos_names,remove,neighbors,weighted);
end
figure(1);
clf
plot(dsd_results_unweighted);hold on;plot(vec_results_unweighted);
title('Unweighted');
axis([0 num_trials 0 1]);
legend('DSD','VEC');
figure(2);
clf
plot(dsd_results_weighted);hold on;plot(vec_results_weighted);
title('Weighted');
axis([0 num_trials 0 1]);
legend('DSD','VEC');