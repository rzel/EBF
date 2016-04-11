function inlier  = BiFun( f1, f2, mu, sigma, fa, fb, thres_spatial)
%BIFUN Summary of this function goes here
%   Detailed explanation goes here

if size(f1,1)> 30
    ind = randsample(size(f1,1), 30);
    f1 = f1(ind,:);
    f2 = f2(ind,:);
end

[G_big, G, lx, ly, X_train] = constructBFdata(f1, f2, mu, sigma);

wx = firBilteralFun(G_big, G, lx, 0.1); 
wy = firBilteralFun(G_big, G, ly, 0.1); 

inlier = preidctBF( fa, fb, X_train, mu, sigma, wx, wy, thres_spatial);

end

