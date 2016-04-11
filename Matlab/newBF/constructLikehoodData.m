function [ mu, sigma, X_train ] = constructLikehoodData( x1, x2 )
%CONSTRUCTLIKEHOODDATA Summary of this function goes here
%   Detailed explanation goes here

X_train =  cat(2, x1, x2);
[X_train, mu, sigma] = featureNormalize(X_train);
X_train(:, 3 : 4) = X_train(:, 3 : 4) - X_train(:, 1 : 2);


end

