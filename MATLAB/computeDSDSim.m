% Convert DSD matrix into a similarity matrix
% Anuththari Gamage, 10/9/2017


filename = '../../Python/DSD/DSD-Source/DSD_Matrix/dsd.txt';
fid = fopen(filename);

%proteinNames = strsplit(fgetl(fid));

%N = length(proteinNames)-1;

DSD = zeros(5884);
n = 0;
while ~feof(fid)
    n = n+1;
%    fscanf(fid, '%s', 1);
    line = fscanf(fid, '%f');
    DSD(n, :) = line';
end
max_val = max(DSD(:));
DSDNorm = DSD./max_val;
fclose(fid);