N = [100,200,500,1000,2000,5000,10000];
c = [2,3,4,5,6,8,10,12,15,20];

for nn = 2:numel(N)
    figure;
    hold on
    yyaxis left
    axis([-inf inf 0 1]);
    errorbar(c,mean(nbt(nn,:,:),3),std(nbt(nn,:,:),0,3))
    errorbar(c,mean(nnbt(nn,:,:),3),std(nnbt(nn,:,:),0,3))
    yyaxis right
    axis([-inf inf 50 100]);
    errorbar(c,mean(cbt(nn,:,:),3),std(cbt(nn,:,:),0,3))
    errorbar(c,mean(cnbt(nn,:,:),3),std(cnbt(nn,:,:),0,3))
    legend({'NMI BT','NMI NBT', 'CCR BT', 'CCR NBT'});
    title(['N = ' num2str(N(nn))]);
    saveas(gcf,sprintf('N%dvariedc1.fig',N(nn)));
    saveas(gcf,sprintf('N%dvariedc1.png',N(nn)));
end
