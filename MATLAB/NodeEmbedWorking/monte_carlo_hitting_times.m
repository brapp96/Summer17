X = [0 .5 0 .5;1/3 0 1/3 1/3;0 1/2 0 1/2;1/3 1/3 1/3 0];
%X = [0 1 0 0;.5 0 .5 0;0 .5 0 .5;0 0 1 0];
X = [0 1/3 1/3 1/3;1/2 0 1/2 0;1/3 1/3 1/3 0;1/2 0 1/2 0];
% X = [0 1 0;.5 0 .5;0 1 0];
X = [0 1 0 0 0 0 0 1;1 0 1 0 0 0 0 0;0 1 0 1 0 0 0 0;0 0 1 0 1 0 0 0;0 0 0 1 0 1 0 0;0 0 0 0 1 0 1 0;0 0 0 0 0 1 0 1;1 0 0 0 0 0 1 0];
num_reps = 100000;
[nX,nY] = size(X);
R = zeros(nX,nY);
fX = cell(1,nX);
numX = zeros(1,nX);
for i = 1:nX
    fX{i} = find(X(i,:));
    numX(i) = numel(fX{i});
end
for i = 1:nX
    for j = 1:nY
        if j == i
            continue;
        end
        result = 0;
        for rep = 1:num_reps
            a = i;
            counter = 0;
            while a ~= j
                counter = counter + 1;
                a = fX{a}(ceil(rand()*numX(a)));
            end
            result = result + counter;
        end
        R(i,j) = result/num_reps;
    end
end
R