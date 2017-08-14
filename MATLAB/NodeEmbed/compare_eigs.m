function compare_eigs(X)
n = size(X,1);
D = diag(sum(X,2));
A = eigs(X,n)'
L = eigs(D-X,n)'
P = eigs(D^-1*X,n)'
almostM = eigs(D^-.5*X*D^-.5,n)'
Lrw = eigs(eye(n) - D^-1*X,n)'
Lsym = eigs(eye(n) - D^-.5*X*D^-.5,n)'
end