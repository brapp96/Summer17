N = [100,200,500,1000,2000,5000,10000];
C = [1,1.5,2,2.5,3,4,5,6,8,10,12,15,20];
K = [2,3];
num_reps = 5;
seed = 45;

ccr_bt = zeros(numel(N),numel(C),num_reps);
ccr_nbt = zeros(numel(N),numel(C),num_reps);
nmi_bt = zeros(numel(N),numel(C),num_reps);
nmi_nbt = zeros(numel(N),numel(C),num_reps);
parfor n = 1:numel(N)
    for c = 1:numel(C)
        for k = 1:numel(K)
            for rep = 1:num_reps
                filename = sprintf('more_graphs/N%d-K%d-c%.1f-la%.1f-iter%d.txt',N(n),K(k),C(c),.9,rep-1);
                fp = fopen(filename,'wb');
                [G,L] = sbm_gen(N(n),K(k),C(c),C(c)/10,seed);
                UG = triu(G);
                size = numel(L);
                for i = 1:size
                    fprintf(fp,'%d ',[i find(UG(i,:))]-1);
                    fprintf(fp,'\n');
                end
                fprintf(fp,'[');
                fprintf(fp,'%d, ',L(1:end-1)-1);
                fprintf(fp,'%d]',L(end)-1);
                fclose(fp);
            end
        end
    end
end