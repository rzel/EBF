function [k, G_ori]=grad_desc_huber_official(x, y, dist_scale, s_scale, thres)



[G] = G_compute_fast(x);
G(G<0.1)=0; %makes it sparse (not utilized here though)
G_ori=G;

M = G * s_scale;
N = ones(size(x,1),1) * s_scale;

n=size(M,2);

try
    [u,s,v]=svd(G);
catch
    
    
    G=G+0.00001*randn(size(G));
    [u,s,v]=svd(G); %sometimes svd goes crasy
end;

G = sqrt(s)*v';


[k,~]=grad_1d(M, N, thres,  G,  1);







