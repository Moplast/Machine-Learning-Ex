function [nn_params, cost] = TrainingNN(X, y, ...
	input_layer_size, hidden_layer_size, num_labels, ...
	lambda, max_iter, initial_nn_params)

% set options for training function fmincg
options = optimset('MaxIter', max_iter);

costFunction = @(p) NNCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);


end