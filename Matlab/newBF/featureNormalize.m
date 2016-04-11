function [X_norm, mu, sigma] = featureNormalize(X, mu, sigma)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.
if nargin == 1
    mu = mean(X);   
end

X_norm = bsxfun(@minus, X, mu);

if nargin == 1
    sigma = std(X_norm);   
end

X_norm = bsxfun(@rdivide, X_norm, sigma);

% ============================================================

end
