% len = [5, 10, 20];
% num_reps = 5;
% BT = cell(numel(len),num_reps);
% NBT = BT;
% [G,~] = import_graph_by_edges(0,0,0);
% for l = 1:numel(len)
%     for rep = 1:num_reps
%         fprintf(1,'length: %d, run: %d\n',len(l),rep);
%         [BT{l,rep},~,~] = node_embed_file(G,0,0,len(l));
%         [NBT{l,rep},~,~] = node_embed_file(G,0,1,len(l));
%     end
% end
% save(['runs/PPI.mat'],'BT','NBT','len');

%%
fp = fopen('../../Python/DSD/DSD-Source/proteinNames_DSD.txt','r');
names = {};
while ~feof(fp)
    names{end+1} = fgetl(fp);
end
fclose(fp);
%%
fp = fopen('../../Python/DSD/DSD-Source/largestCluster2.txt','r');
largeclusternames = {};
while ~feof(fp)
    largeclusternames{end+1} = fgetl(fp); %#ok<*SAGROW>
end
fclose(fp);

%%
k = 18;
Lbt = BT;
Lnbt = BT;
inter = names;
for i = 6%:15
    Lbt{i} = kmeans(BT{i},k);
    Lnbt{i} = kmeans(NBT{i},k);
    [~,ind] = sort(hist(Lbt{i},1:k));
    fp = fopen('VEClargestCluster1.txt','w');
    words=join(names(Lbt{i}==ind(end)),' ');
    fprintf(fp,'%s\n',words{:});
    unix('sed -e "s/\s\+/\n/g" VEClargestCluster1.txt > VEClargestCluster.txt');
    inter = intersect(inter,names(Lbt{i}==ind(end)));
end




