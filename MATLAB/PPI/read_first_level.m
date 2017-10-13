function [names,data] = read_first_level(filename)
    fp = fopen(filename,'r');
    names = {};
    data = {};
    while ~feof(fp)
        line = strsplit(fgetl(fp));
        names{end+1} = line{1};
        data{end+1} = str2num(char(line(2:end-1)));
    end
end