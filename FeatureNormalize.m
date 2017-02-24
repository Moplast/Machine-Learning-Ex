function [X_norm, mu, sigma] = FeatureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1.

% Init
X_norm = X;
mu = zeros(1, size(X, 2));      % mean value 
sigma = zeros(1, size(X, 2));   % standard deviation

% mean and std
mu = mean(X);
stdv = std(X);

% normlization
X_norm = (X - repmat(mu, size(X, 1), 1)) ./ repmat(stdv, size(X, 1), 1);

% ============================================================

end
