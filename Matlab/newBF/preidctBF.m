function inlier  = preidctBF( fa, fb, X_train, mu, sigma, wx, wy, thres_spatial)
%GETTESTDATA Summary of this function goes here
%   Detailed explanation goes here

X_test = cat(2, fa, fb);
X_test = featureNormalize(X_test, mu(1 : 4), sigma(1 : 4));
lx = X_test(:, 3);
ly = X_test(:, 4);
X_test(:, 3 : 4) = X_test(:, 3 : 4) - X_test(:, 1 : 2);


G = G_compute_fast(X_test,X_train,1);
G_1 = cat(2, G, ones(size(X_test, 1), 1));
G1 = bsxfun(@times, G_1, X_test(:, 1));
G2 = bsxfun(@times, G_1, X_test(:, 2));
G_big = cat(2, G1, G2, G_1);


e1 = G_big * wx - lx;
e2 = G_big * wy - ly;
e = e1.^2 + e2.^2;
inlier = e < thres_spatial;

end

