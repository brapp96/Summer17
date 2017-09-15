% function L=create_combinatorial_lap(A)
%
% creates the Laplacian graph of a graph
% Input:
% - A the adjacency matrix (possibly weighted) of the graph. 
% A should be either symmetrical.
% Output:
% - L the Laplacian matrix
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

function L=create_combinatorial_lap(A)

% if istriu(A)
%     A=A+A';
% elseif ~issymmetric(A)
%    error('create_laplaciangraph: A should be tri sup or sym');
% end

L=diag(sum(A))-A; 

