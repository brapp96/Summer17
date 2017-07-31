N = 100;
load('nmi_ccr_07-28-17_10:54:13.mat');

figure;
hold on
yyaxis left
errorbar(c,mean(nmi_bt,3),std(nmi_bt,0,3))
errorbar(c,mean(nmi_nbt,3),std(nmi_nbt,0,3))
yyaxis right
errorbar(c,mean(ccr_bt,3),std(ccr_bt,0,3))
errorbar(c,mean(ccr_nbt,3),std(ccr_nbt,0,3))
legend({'NMI BT','NMI NBT', 'CCR BT', 'CCR NBT'});
title(['Backtracking vs. Nonbacktracking, N = ' N]);
saveas(gcf,sprintf('figs/N%dvariedc.fig',N));
saveas(gcf,sprintf('figs/N%dvariedc.png',N));