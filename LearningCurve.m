function [error_train, error_cv] = LearningCurve(X, y, ...
	X_cv, y_cv, ...
	num_labels, ...
	input_layer_size, hidden_layer_size, ...
	lambda, max_iter, epsilon_init)


%LEARNINGCURVE Generates the train and cross validation set errors needed 
%to plot a learning curve
%   [error_train, error_val] = ...
%       LEARNINGCURVE(X, y, Xval, yval, lambda) returns the train and
%       cross validation set errors for a learning curve. In particular, 
%       it returns two vectors of the same length - error_train and 
%       error_val. Then, error_train(i) contains the training error for
%       i examples (and similarly for error_val(i)).

% Initialization
m_train = size(X, 1);
m_cv = size(X_cv, 1);
error_train = zeros(floor(m_train/2), 1);
error_cv = zeros(floor(m_train/2), 1);

% initialize nn params
initial_nn_params = InitNNTheta( input_layer_size, hidden_layer_size, num_labels, ...
	epsilon_init);



for i = 1 : 2: m_train
	
	if (mod(i, 20)==1)
		fprintf('Completed: %d / %d...\n', floor(i/2), floor(m_train/2));
	end
	
	% update training examples
	X_train = X(1:i, :);
	y_train = y(1:i);
	
	[nn_params, cost] = TrainingNN(X_train, y_train, ...
	input_layer_size, hidden_layer_size, num_labels, ...
	lambda, max_iter, initial_nn_params);
	
	% update training set results and validation results 
	%h_train = ActivationFun([ones(i, 1) ActivationFun([ones(i, 1) X_train] * Theta1')] * Theta2');
	%h_cv = ActivationFun([ones(m_cv, 1) ActivationFun([ones(m_cv, 1) X_cv] * Theta1')] * Theta2');	
	
	% calculate error of training examples and validation set
	error_train((i+1)/2) = NNCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X_train, y_train, lambda);
	error_cv((i+1)/2) = NNCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X_cv, y_cv, lambda);
	
end

% -------------------------------------------------------------

% =========================================================================

end
