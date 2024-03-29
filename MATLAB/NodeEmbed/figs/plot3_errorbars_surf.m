function [h]=plot3_errorbars_surf(x, y, z, e)
% this matlab function plots 3d data using the plot3 function
% it adds vertical errorbars to each point symmetric around z
% I experimented a little with creating the standard horizontal hash
% tops the error bars in a 2d plot, but it creates a mess when you 
% rotate the plot
%
% x = xaxis, y = yaxis, z = zaxis, e = error value

% create the standard 3d scatterplot
hold off;
h=plot3(x, y, z, '.k');

% looks better with large points
set(h, 'MarkerSize', 25);
hold on

% now draw the vertical errorbar for each point
for i=1:length(x)
        xV = [x(i); x(i)];
        yV = [y(i); y(i)];
        zMin = z(i) + e(i);
        zMax = z(i) - e(i);

        zV = [zMin, zMax];
        % draw vertical error bar
        h=plot3(xV, yV, zV, '-k');
        set(h, 'LineWidth', 2);
end

% now we want to fit a surface to our data
% the  0.25 and 0.1 define the density of the fit surface
% adjust them to your liking
tt1=[floor(min(min(x))):0.25:max(max(x))];
tt2=[floor(min(min(y))):0.1:max(max(y))];

% prepare for fitting the surface
[xg,yg]=meshgrid(tt1,tt2);

% fit the surface to the data; 
% matlab has several choices for the fit;  below is "linear"
zg=griddata(x, y, z, xg,yg,'linear');
% draw the mesh on our plot
surf(xg,yg,zg), xlabel('x axis'), ylabel('y axis'), zlabel('z axis')