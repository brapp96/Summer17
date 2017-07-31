N = 100;%[100,200,500,1000,2000,5000,10000];
c = [2,3,4,5,6,8,10,12,15,20];
iters = 0:4;
num_reps = 2;
num_total = num_reps*(max(iters)+1);

ccr_bt = zeros(numel(N),numel(c),num_total);
ccr_nbt = ccr_bt;
nmi_bt = ccr_bt;
nmi_nbt = ccr_bt;
logfile = fopen('test.log','wb');
for i = 1:numel(N)
    for j = 1:numel(c)
        for iter = iters
            [G,L] = import_graph_by_edges(N(i),2,c(j),.9,iter);
            for rep = 1:num_reps
                fprintf(logfile,'N = %d, c = %d, iter %d, rep %d\n',N(i),c(j),iter,rep);
                ind = num_reps*iter+rep;
                [~,ccr_bt(i,j,ind),nmi_bt(i,j,ind)] = node_embed_file(G,L,0);
                [~,ccr_nbt(i,j,ind),nmi_nbt(i,j,ind)] = node_embed_file(G,L,1);
            end
        end
    end
end
fclose(logfile);
save(['figs/nmi_ccr_' datestr(clock,'mm-dd-yy_HH:MM:SS') '.mat'],'nmi_bt','nmi_nbt','ccr_bt','ccr_nbt');
