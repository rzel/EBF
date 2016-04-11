function inlier = LikehoodFilter( x1, x2, w, X_train, mu, sigma, inlier_thres)
%LIKEHOODFILTER Summary of this function goes here
%   Detailed explanation goes here
X_test = cat(2, x1, x2);
X_test = featureNormalize(X_test, mu, sigma);
X_test(:, 3 : 4) = X_test(:, 3 : 4) - X_test(:, 1 : 2);
G = G_compute_fast(X_test, X_train, 1);
inlier = G * w > inlier_thres;

end

