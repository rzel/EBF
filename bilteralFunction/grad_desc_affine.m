function [w, a0, G_ori]=grad_desc_affine(x, y, dist_scale, s_scale, impact_idx, thres)


[G]=G_compute_fast(x);
G_ori=G;


M=zeros(size(x,1), 3*size(x,1)+3);
N=zeros(size(x,1),1);
num=size(G,1);

%construct M
M(:,1:num) = bsxfun(@times, G, x(:, 1));
M(:,num+1:2*num) = bsxfun(@times, G, x(:, 2));
M(:,2*num+1:3*num) = G;
M(:,3*num+1) = x(:,1);
M(:,3*num+2) = x(:,2);
M(:,3*num+3) = 1;




M = bsxfun(@times, M, impact_idx) * s_scale;
N = y' .* impact_idx *s_scale;

n = size(M,2);

[~,s,v]=svd(G);
G = sqrt(s)*v';


[k,~]=grad_aff2(M, N, thres,  G,  1);

num=size(G,1);
w=[k(1:num) k(num+1:2*num) k(2*num+1:3*num)];
a0=[k(3*num+1);k(3*num+2);k(3*num+3)];



