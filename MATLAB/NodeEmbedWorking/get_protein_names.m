function names = get_protein_names(filename)
    names = {};
    fp = fopen(filename,'r');
    while ~feof(fp)
        names{end+1} = fgetl(fp);
    end
    fclose(fp);
end