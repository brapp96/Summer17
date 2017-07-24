% Generate graphs of Vec performance results, evaluated using CCR and NMI
% 7/13/2017 Anu Gamage

function plot_metrics(ccr_brw, ccr_nbrw, nmi_brw, nmi_nbrw)

x = 1:length(ccr_brw);
figure

% Plot CCR results
subplot(2,1,1)
plot(x, ccr_brw,'-o')
hold on
plot(x, ccr_nbrw,'-o')
xlabel('#tests')
ylabel('CCR')
ylim([30,100])
title('CCR Results')
legend('Backtracking', 'Non-backtracking')

% Plot NMI results
subplot(2,1,2)
plot(x, nmi_brw,'-o')
hold on
plot(x, nmi_nbrw,'-o')
xlabel('#tests')
ylabel('NMI')
ylim([0,1])
title('NMI Results')
legend('Backtracking', 'Non-backtracking')

end
