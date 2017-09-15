% function L=create_laplacian(A,lapversion)
%
% creates the Laplacian matrix of a graph represented by its adjacency matrix A. 
% lapversion should be a string equal to 'combinatorial' if you want L=D-A; 
% or 'normalized' if you want the normalized version of the laplacian L=I-D^(-1/2)*A*D^(-1/2); 
% where D is the diagonal matrix of the degrees. 
% A should be symmetrical.
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

function L=create_laplacian(A,lapversion)

if strcmp(lapversion, 'combinatorial')
    L=create_combinatorial_lap(A);
elseif strcmp(lapversion, 'normalized')   
    L=create_normalized_lap(A);
else error('Laplacian type should be ''combinatorial'' or ''normalized''');
end

