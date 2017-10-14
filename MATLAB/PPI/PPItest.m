len = [30 45 60];
num_reps = 2;
BT = cell(numel(len),num_reps);
NBT = BT;
[G,~] = import_graph_by_edges(0,0,0);
for l = 1:numel(len)
    for rep = 1:num_reps
        fprintf(1,'length: %d, run: %d\n',len(l),rep);
        [BT{l,rep},~,~] = node_embed_file(G,0,0,len(l));
        [NBT{l,rep},~,~] = node_embed_file(G,0,1,len(l));
    end
end
fclose(fp);