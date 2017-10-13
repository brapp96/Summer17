% Convert DSD matrix into a similarity matrix
% Anuththari Gamage, 10/9/2017

filename = 'Python/DSD/DSD-Source/DSD_Matrix/data/results_converged.DSD1';
fid = fopen(filename,'r');

proteinNames = strsplit(fgetl(fid));

N = length(proteinNames)-1;

DSD = zeros(N);
n = 0;
while ~feof(fid)
    n = n+1;
    fscanf(fid, '%s', 1);
    line = fscanf(fid, '%f');
    DSD(n,:) = line';
end
max_val = max(DSD(:));
DSDNorm = DSD./max_val;
fclose(fid);