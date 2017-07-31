N = [100,200,500,1000,2000,5000,10000];
c = [2,3,4,5,6,8,10,12,15,20];
iters = 0:4;
num_reps = 5;
num_total = num_reps*(max(iters)+1);

cbt = zeros(numel(N),numel(c),num_total);
cnbt = cbt;
nbt = cbt;
nnbt = cbt; 
logfile = fopen('test.log','wb');
for i = 1:numel(N)
    for j = 1:numel(c)
        for iter = iters
            [G,L] = import_graph_by_edges(N(i),2,c(j),.9,iter);
            for rep = 1:num_reps
                fprintf(logfile,'N = %d, c = %d, iter %d, rep %d\n',N(i),c(j),iter,rep);
                ind = num_reps*iter+rep;
                [~,cbt(i,j,ind),nbt(i,j,ind)] = node_embed_file(G,L,0);
                [~,cnbt(i,j,ind),nnbt(i,j,ind)] = node_embed_file(G,L,1);
            end
        end
    end
end
fclose(logfile);
save(['figs/nmi_ccr_' datestr(clock,'mm-dd-yy_HH:MM:SS') '.mat'],'nbt','nnbt','cbt','cnbt');
% for nn = 2:numel(N)
%     figure;
%     hold on
%     yyaxis left
%     axis([-inf inf 0 1]);
%     errorbar(c,mean(nbt(nn,:,:),3),std(nbt(nn,:,:),0,3))
%     errorbar(c,mean(nnbt(nn,:,:),3),std(nnbt(nn,:,:),0,3))
%     yyaxis right
%     axis([-inf inf 50 100]);
%     errorbar(c,mean(cbt(nn,:,:),3),std(cbt(nn,:,:),0,3))
%     errorbar(c,mean(cnbt(nn,:,:),3),std(cnbt(nn,:,:),0,3))
%     legend({'NMI BT','NMI NBT', 'CCR BT', 'CCR NBT'});
%     title(['N = ' num2str(N(nn))]);
%     saveas(gcf,sprintf('N%dvariedc1.fig',N));
%     saveas(gcf,sprintf('N%dvariedc1.png',N));
% end
