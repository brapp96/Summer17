function print_graphs(filename,var_type) %#ok<*NODEF>
% For printing data from any set of comparable RW and NBRW runs
load(filename);
C = c;
cd(userpath);
cd NodeEmbed/
for k = 1:numel(K)
    for l = 1:numel(len)
        if var_type == 'c'
            for n = 1:numel(N)
                figure;
                hold on
                yyaxis left
                axis([-inf inf 0 1]);
                errorbar(c,mean(nmi_bt(n,:,k,l,:),5),std(nmi_bt(n,:,k,l,:),0,5))
                errorbar(c,mean(nmi_nbt(n,:,k,l,:),5),std(nmi_nbt(n,:,k,l,:),0,5))
                ylabel('NMI');
                yyaxis right
                axis([-inf inf 100/K(k) 100]);
                errorbar(c,mean(ccr_bt(n,:,k,l,:),5),std(ccr_bt(n,:,k,l,:),0,5))
                errorbar(c,mean(ccr_nbt(n,:,k,l,:),5),std(ccr_nbt(n,:,k,l,:),0,5))
                ylabel('CCR');
                xlabel('average degree within cluster');
                legend({'NMI BT','NMI NBT', 'CCR BT', 'CCR NBT'},'Location','best');
                title(sprintf('%d nodes, %d clusters, %d step walk',N(n),K(k),len(l)));
                saveas(gcf,sprintf('figs/N%dK%dlen%d.png',N(n),K(k),len(l)));
            end
        elseif var_type == 'N'
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
end