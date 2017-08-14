function print_graphs_N(filename) %#ok<*NODEF>
% For printing data from any set of comparable RW and NBRW runs
load(filename);
C = c;
for k = 1:numel(K)
    for l = 1:numel(len)
        for c = 1:numel(C)
            figure;
            hold on
            yyaxis left
            axis([-inf inf 0 1]);
            errorbar(N,mean(nmi_bt(:,c,k,l,:),5),std(nmi_bt(:,c,k,l,:),0,5))
            errorbar(N,mean(nmi_nbt(:,c,k,l,:),5),std(nmi_nbt(:,c,k,l,:),0,5))
            ylabel('NMI');
            yyaxis right
            axis([-inf inf 100/K(k) 100]);
            errorbar(N,mean(ccr_bt(:,c,k,l,:),5),std(ccr_bt(:,c,k,l,:),0,5))
            errorbar(N,mean(ccr_nbt(:,c,k,l,:),5),std(ccr_nbt(:,c,k,l,:),0,5))
            ylabel('CCR');
            xlabel('number of nodes in graph');
            set(gca,'XScale','log');
            legend({'NMI BT','NMI NBT', 'CCR BT', 'CCR NBT'},'Location','best');
            title(sprintf('average degree %d, %d clusters, %d step walk',C(c),K(k),len(l)));
            saveas(gcf,sprintf('figs/C%dK%dlen%d.png',C(c),K(k),len(l)));
        end
    end
end
end