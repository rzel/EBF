function [G]=G_compute_fast(X, Y, dist_scale)
% Gaussian similarity Matrix
%G=G.^2;

if nargin == 3
    G = exp(- dist_scale * vl_alldist2(X,Y) );
end
   
if nargin == 1
    XX=sum(X.*X,2);
    K = bsxfun(@plus,-2*(X*X'), bsxfun(@plus, XX, XX'));
    G = exp(- K );
end

end