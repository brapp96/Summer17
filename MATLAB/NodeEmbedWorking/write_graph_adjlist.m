function write_graph_adjlist(G,L,filename)
    fp = fopen(filename,'wb');
    UG = triu(G);
    size = numel(L);
    for i = 1:size
        fprintf(fp,'%d ',[i find(UG(i,:))]-1);
        fprintf(fp,'\n');
    end
    fprintf(fp,'[');
    fprintf(fp,'%d, ',L(1:end-1)-1);
    fprintf(fp,'%d]',L(end)-1);
    fclose(fp);
end