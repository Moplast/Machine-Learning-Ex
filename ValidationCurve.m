function [lambda_vec, error_train, error_cv] = ...
    ValidationCurve(X, y, X_cv, y_cv,...
	num_labels,...
	input_layer_size, hidden_layer_size,...
	epsilon_init, max_iter)
%VALIDATIONCURVE Generate the train and validation errors needed to
%plot a validation curve that we can use to select lambda
%   [lambda_vec, error_train, error_val] = ...
%       VALIDATIONCURVE(X, y, Xval, yval) returns the train
%       and validation errors (in error_train, error_val)
%       for different values of lambda. You are given the training set (X,
%       y) and validation set (Xval, yval).

% Selected values of lambda 
lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]';
error_train = zeros(length(lambda_vec), 1);
error_cv = zeros(length(lambda_vec), 1);

% set options for training function fmincg
options = optimset('MaxIter', max_iter);

% initial parameters
initial_Theta1 = RandInitializeWeights(input_layer_size, hidden_layer_size, epsilon_init);
initial_Theta2 = RandInitializeWeights(hidden_layer_size, num_labels, epsilon_init);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

for i = 1: length(lambda_vec)
	
	fprintf('Completed: %d / %d...\n', i, length(lambda_vec));

	lambda = lambda_vec(i);
	
	% Train: train NN regularization parameters
	costFunction = @(p) NNCostFunction(p, ...
									   input_layer_size, ...
									   hidden_layer_size, ...
									   num_labels, X, y, lambda);
	[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);
	
	% update training set results and validation results 
	%h_train = ActivationFun([ones(i, 1) ActivationFun([ones(i, 1) X_train] * Theta1')] * Theta2');
	%h_cv = ActivationFun([ones(m_cv, 1) ActivationFun([ones(m_cv, 1) X_cv] * Theta1')] * Theta2');	
	
	% calculate error of training examples and validation set
	error_train(i) = NNCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda);
	error_cv(i) = NNCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X_cv, y_cv, lambda);
		
end



% =========================================================================

end
