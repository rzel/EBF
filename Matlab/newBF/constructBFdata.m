function [ G_big, G, lx, ly, X_train ] = constructBFdata( f1, f2, mu, sigma)
%CONSTRUCTBFDATA Summary of this function goes here
%   Detailed explanation goes here
X_train = cat(2, f1, f2);
X_train = featureNormalize(X_train, mu(1 : 4), sigma(1 : 4)); 
lx = X_train(:, 3);
ly = X_train(:, 4);
X_train(:, 3 : 4) = X_train(:, 3 : 4) - X_train(:, 1 : 2);

m = size(X_train, 1);
x = X_train(:, 1); 
y = X_train(:, 2);
G = G_compute_fast(X_train);
G_1 = cat(2, G, ones(m, 1));
G1 = bsxfun(@times, G_1, x);
G2 = bsxfun(@times, G_1, y);
G_big = cat(2, G1, G2, G_1);


end

