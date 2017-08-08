function print_graphs(filename,var) %#ok<*NODEF>
% For printing data from any set of comparable RW and NBRW runs
load(filename);
switch var
    case {'c','C'}
        if numel(c) == 1
            error('c should be greater than 1');
        end
        len = 3;
        for nn = 1:numel(N)
            figure;
            hold on
            yyaxis left
            axis([-inf inf 0 1]);
            errorbar(c,mean(nmi_bt(nn,:,1,len,:),5),std(nmi_bt(nn,:,1,len,:),0,5))
            errorbar(c,mean(nmi_nbt(nn,:,1,len,:),5),std(nmi_nbt(nn,:,1,len,:),0,5))
            yyaxis right
            axis([-inf inf 50 100]);
            errorbar(c,mean(ccr_bt(nn,:,1,len,:),5),std(ccr_bt(nn,:,1,len,:),0,5))
            errorbar(c,mean(ccr_nbt(nn,:,1,len,:),5),std(ccr_nbt(nn,:,1,len,:),0,5))
            legend({'NMI BT','NMI NBT', 'CCR BT', 'CCR NBT'},'Location','southeast');
            title(['N = ' num2str(N(nn))]);
            saveas(gcf,sprintf('N%dvariedc.fig',N(nn)));
            saveas(gcf,sprintf('N%dvariedc.png',N(nn)));
        end
    case {'n','N'}
        if numel(N) == 1
            error('N should be greater than 1');
        end
        for cc = 1:numel(c)
            figure;
            hold on
            yyaxis left
            axis([-inf inf 0 1]);
            errorbar(N,mean(nmi_bt(:,cc,:),3),std(nmi_bt(:,cc,:),0,3))
            errorbar(N,mean(nmi_nbt(:,cc,:),3),std(nmi_nbt(:,cc,:),0,3))
            yyaxis right
            axis([-inf inf 50 100]);
            errorbar(N,mean(ccr_bt(:,cc,:),3),std(ccr_bt(:,cc,:),0,3))
            errorbar(N,mean(ccr_nbt(:,cc,:),3),std(ccr_nbt(:,cc,:),0,3))
            legend({'NMI BT','NMI NBT', 'CCR BT', 'CCR NBT'},'Location','southeast');
            title(['c = ' num2str(c(cc))]);
            saveas(gcf,sprintf('c%dvariedN.fig',c(cc)));
            saveas(gcf,sprintf('c%dvariedN.png',c(cc)));
        end
    otherwise
        error('Must be one of "N","n","C","c".');
end
end