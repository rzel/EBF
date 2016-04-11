function [G]=G_compute_fast(X, Y, dist_scale)
% Gaussian similarity Matrix
%G=G.^2;

if nargin == 3
    G = exp(- dist_scale * vl_alldist(X',Y') );
end
   
if nargin == 1
     G = exp(-vl_alldist(X'));
end

end