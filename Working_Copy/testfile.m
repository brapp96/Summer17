N = [100,200,500,1000,2000,5000,10000];
K = [2,3,5];
c = [2,3,4,5,6,8,10,12,15,20];
lambda = .9;
num_reps = 20;
quiet = 0;
len = [5,10,20];
seed = 44;

total_time = sum(N)*sum(K)*sum(c)*num_reps*numel(len);
curr_time = 0;
tic;
ccr_bt = zeros(numel(N),numel(c),numel(K),numel(len),num_reps);
ccr_nbt = zeros(numel(N),numel(c),numel(K),numel(len),num_reps);
nmi_bt = zeros(numel(N),numel(c),numel(K),numel(len),num_reps);
nmi_nbt = zeros(numel(N),numel(c),numel(K),numel(len),num_reps);
for i = 1:numel(N)
    for j = 1:numel(c)
        for k = 1:numel(K)
            seed = seed+1;
            %[G,L] = import_graph_by_edges(N(i),K(k),c(j)); for reproducibility
            [G,L] = sbm_gen(N(i),K(k),c(j),c(j)*(1-lambda),seed);
            filename = sprintf('more_graphs/N%d-K%d-c%.1f.txt',N(i),K(k),c(j));
            write_graph_adjlist(G,L,filename);
            for l = 1:numel(len)
                if ~quiet
                    fprintf('N = %d, K = %d, c = %d, len = %d\n',N(i),K(k),c(j),len(l));
                end
                for rep = 1:num_reps
                    [~,ccr_bt(i,j,k,l,rep),nmi_bt(i,j,k,l,rep)] = node_embed_file(G,L,0,len(l));
                    [~,ccr_nbt(i,j,k,l,rep),nmi_nbt(i,j,k,l,rep)] = node_embed_file(G,L,1,len(l));
                    curr_time = curr_time+N(i)*c(j)*K(k);
                    if ~quiet
                        fprintf('%4.2f%% done; %4.2fs, %4.2fs estimated\n',100*curr_time/total_time,toc,toc/curr_time*total_time);
                    end
                end
            end
        end
    end
end
save(['runs/nmi_ccr_' datestr(clock,'mm-dd-yy_HH:MM:SS') '.mat'],'nmi_bt','nmi_nbt','ccr_bt','ccr_nbt','N','K','c','len');
% for nn = 1:numel(N)
%     figure;
%     hold on
%     yyaxis left
%     axis([-inf inf 0 1]);
%     errorbar(c,mean(nmi_bt(nn,:,:),3),std(nmi_bt(nn,:,:),0,3))
%     errorbar(c,mean(nmi_nbt(nn,:,:),3),std(nmi_nbt(nn,:,:),0,3))
%     yyaxis right
%     axis([-inf inf 50 100]);
%     errorbar(c,mean(ccr_bt(nn,:,:),3),std(ccr_bt(nn,:,:),0,3))
%     errorbar(c,mean(ccr_nbt(nn,:,:),3),std(ccr_nbt(nn,:,:),0,3))
%     legend({'NMI BT','NMI NBT', 'CCR BT', 'CCR NBT'});
%     title(['N = ' num2str(N(nn))]);
%     saveas(gcf,sprintf('N%dvariedc.fig',N(nn)));
%     saveas(gcf,sprintf('N%dvariedc.png',N(nn)));
% end
