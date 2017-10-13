n = 10;
a = zeros(n,2);
b = zeros(n,2);
for ii = 1:n
    [a(ii,1),~] = nearest_neighbor(DSDNorm,excessnames2,FLdata,FLnames,50,500,1);
    [a(ii,2),~] = nearest_neighbor(DSDNorm,excessnames2,FLdata,FLnames,50,500,0);
    [b(ii,1),~] = nearest_neighbor(GG,excessnames2,FLdata,FLnames,50,500,1);
    [b(ii,2),~] = nearest_neighbor(GG,excessnames2,FLdata,FLnames,50,500,0);
end
figure(1);
clf
plot(a(:,1));hold on;plot(b(:,1));
title('Unweighted');
axis([0 n 0 1]);
legend('DSD','VEC');
figure(2);
clf
plot(a(:,2));hold on;plot(b(:,2));
title('Weighted');
axis([0 n 0 1]);
legend('DSD','VEC');