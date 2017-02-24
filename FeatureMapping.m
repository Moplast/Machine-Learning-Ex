function [X_map, input_layer_size] = FeatureMapping(X)

input_layer_size = 3;

X_map = zeros(size(X,1), input_layer_size);

X_map(:, 1:size(X, 2)) = X;
% X_map(:, 4) = X(:, 1).*2;
% X_map(:, 5) = X(:, 1).*3;

% X_map(:, 5) = X(:, 1).*2;
% X_map(:, 6) = X(:, 2).*2;
% 
% X_map(:, 7) = X(:, 1).*X(:, 2);
% X_map(:, 8) = X(:, 1).*X(:, 3);
% X_map(:, 9) = X(:, 2).*X(:, 3);


end