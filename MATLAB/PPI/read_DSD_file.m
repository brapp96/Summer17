function [DSD,protein_names] = read_DSD_file(filename)
    fp = fopen(filename,'r');
    protein_names = strsplit(fgetl(fp));
    N = length(protein_names)-1;
    DSD = zeros(N);
    n = 0;
    while ~feof(fp)
        n = n+1;
        fscanf(fp, '%s', 1);
        line = fscanf(fp, '%f');
        DSD(n,:) = line';
    end
    fclose(fp);
end