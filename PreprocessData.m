function [ X, y, num_labels, input_layer_size ] = PreprocessData(X, Y, file_col,...
	mapping_option, normalization_option)


% initial
input_layer_size = size(X, 2);

% output class processing
% deal with first column first
[ y, yMap ] = Numerial2Class(Y(:, file_col)); 

% update output label number
num_labels = length(yMap);

% feature mapping
if mapping_option == 1
	[X, input_layer_size] = FeatureMapping(X);
end

% feature normalization
if normalization_option == 1
	[X, ~, ~] = FeatureNormalize(X);
end


end