% function Ln=create_normalized_lap(A)
%
% creates the normalized Laplacian graph of a graph.
% Input:
% - A the adjacency matrix (possibly weighted) of the graph. 
% A should be either symmetrical. 
% Output:
% - Ln the normalized Laplacian matrix
%
% Copyright (C) 2016 Nicolas Tremblay, Gilles Puy.
% This file is part of the CSCbox (Compressive Spectral Clustering toolbox)
%
% The CSCbox is free software: you can redistribute it and/or modify
% it under the terms of the GNU Affero General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
%
% The CSCbox is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU Affero General Public License
% along with this program.  If not, see <http://www.gnu.org/licenses/>.
%
% If you use this toolbox please kindly cite
%     N. Tremblay, G. Puy, R. Gribonval and P. Vandergheynst.
%     Compressive Spectral Clustering.
%     Proc. of the 33rd Int. Conf. on Machine Learning (ICML 2016), NY, USA
%     [ArXiv:1602.02018] 


function Ln=create_normalized_lap(A)

% if istriu(A)%istrisup(A)
%     A=A+A';
% elseif ~issymmetric(A)
%     error('create_normlaplaciangraph: A should be tri sup or sym');
% end

DEG_dem=spdiags(1./sqrt(sum(A))',0,size(A,1),size(A,2));
%DEG_dem=sparse(size(A,1),size(A,2));
%diag(DEG_dem)=1./sqrt(sum(A));

Q=speye(length(A))-DEG_dem*A*DEG_dem;

% make sure the Laplacian is exactly symmetrical:
Qbis=triu(Q,1);
Qbis=Qbis+Qbis';
Ln=Qbis+spdiags(diag(Q),0,size(A,1),size(A,2));
Ln(isnan(Ln)) = 0 ;
