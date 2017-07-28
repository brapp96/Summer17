N = 2000;
c = [2,3,4,5,6,8,10,12,15,20];
iter = 0:4;
num_reps = 5;
num_total = num_reps*(max(iter)+1);

cbt = zeros(num_total,numel(c));
cnbt = zeros(num_total,numel(c));
nbt = zeros(num_total,numel(c));
nnbt = zeros(num_total,numel(c));
for i = 1:numel(c)
    for iter = 0:4
        [G,L] = import_graph_by_edges(N,2,c(i),.9,iter);
        for rep = 1:num_reps
            fprintf('c = %d, iter %d, rep %d\n',c(i),iter,rep);
            ind = num_reps*(iter)+rep;
            tic
            [~,cbt(ind,i),nbt(ind,i)] = node_embed_file(G,L,0);
            toc
            [~,cnbt(ind,i),nnbt(ind,i)] = node_embed_file(G,L,1);
            toc
        end
    end
end
figure;
hold on
yyaxis left
errorbar(c,mean(nbt),std(nbt))
errorbar(c,mean(nnbt),std(nnbt))
yyaxis right
errorbar(c,mean(cbt),std(cbt))
errorbar(c,mean(cnbt),std(cnbt))
legend({'NMI BT','NMI NBT', 'CCR BT', 'CCR NBT'});
title('Variation of c');
saveas(gcf,sprintf('N%dvariedc.png',N));
