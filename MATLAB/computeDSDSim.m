% Convert DSD matrix into a similarity matrix
% Anuththari Gamage, 10/9/2017


filename = 'dsd.txt';
fid = fopen(filename);

sigma = 1e2;
proteinNames = strsplit(fgetl(fid));

N = length(proteinNames)-1;

DSD = zeros(N,N);
n = 0;
while ~feof(fid)
    n = n+1;
    fscanf(fid, '%s', 1);
    line = fscanf(fid, '%f');
    DSD(n, :) = line';
end
max_val = max(DSD(:));
DSD = DSD./max_val;
fclose(fid);
fclose(fout);