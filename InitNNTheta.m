function initial_nn_params = InitNNTheta( input_layer_size, ...
	hidden_layer_size, num_labels, epsilon_init)

initial_Theta1 = RandInitializeWeights(input_layer_size, hidden_layer_size, ...
	epsilon_init);
initial_Theta2 = RandInitializeWeights(hidden_layer_size, num_labels, ...
	epsilon_init);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

end