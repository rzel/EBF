function [T_a, T_b, w, Vec]=matches2points_affine_reguess(fa, fb, aff, thres)

% spatial coordinates
xa = fa(:,1);
ya = fa(:,2);
xb = fb(:,1);
yb = fb(:,2);

%normalize points so constant parameters can be used
[a_points, T_a]=normalise2dpts(cat(2, xa, ya, ones(length(xa),1))');
[b_points, T_b]=normalise2dpts(cat(2, xb, yb, ones(length(xa),1))');
a_points=a_points';
b_points=b_points';

matched=cat(2, xa, ya, xb, yb, aff);
y=ones(size(matched,1), 1);

Vec=cat(2, a_points(:,1:2), b_points(:,1:2)-a_points(:,1:2),aff);
% compute the curve fit, thres is the threshold for the huber function

%[w, ~]=grad_desc_huber_official(cat(2, a_points(:,1:2), (b_points(:,1:2)-a_points(:,1:2)),aff),y,1, 1, thres);
[w, ~]=grad_desc_huber_official(Vec,y,1, 1, thres);

