function [J grad] = NNCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 

% Initialization
m = size(X, 1);
J = 0;

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
			 
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
%% Part 1: J part----------------------------------------------
% #Forward propagation#

ones_column = ones(m, 1);
% a1[ m * features + 1]
a1 = [ones_column X];
% z1[ m * hidden_layer_size]
z2 = a1 * Theta1';
% a2[ m * hidden_layer_size + 1]
a2 = [ones_column ActivationFun(z2)];
% z2[ m * ouput_layer_size]
z3 = a2 * Theta2';
% a3[ m * output_layer_size]
a3 = ActivationFun(z3);
h = a3;

% recode h and y
for i = 1 : m	
	Y = zeros(1, num_labels);
	H = h(i, :);
	Y(y(i)) = 1;
	J = J + sum(log(H).*(-Y) - log(1-H).*(1-Y));
end

J = J / m ;

%% Part 2: Gradient part----------------------------------------------
% #Backpropagation#
delta_1 = zeros(size(Theta1));
delta_2 = zeros(size(Theta2));
% recode h and y
for i = 1 : m	
	Y = zeros(1, num_labels);
	Y(y(i)) = 1;
	
	% Step 1: set the input layer's values (a(1)) 
	%        to the i-th training example
	a_1 = a1(i, :); % [1* features+1]
	a_2 = a2(i, :); % [1* hidden_layer_size + 1]
	a_3 = a3(i, :); % [1* output_size]
	z_2 = z2(i, :); % [1 * hidden_layer_size]
	
	% Step 2: for each output unit k in layer 3, set
	% error3 [output_size * 1]
	error_3 = (a_3 - Y)'; 
	
	% Step 3: for the hidden layer l = 2, set
	% error2 [hidden_size * 1]
	% Theta2'[hidden_size+1 * output_size] error3 [output_size * 1] z_2 [1 * output_size]
	temp = Theta2' * error_3;
	[~, gd] = ActivationFun(z_2');
	error_2 = temp(2 : end) .* gd;
	
	% Step 4: Accumulate the gradient from this example using the
	% following formula. Note that to skip error0_2
	% delta_2 [output_size * hidden_size+1] error3 [output_size * 1] a_2 [1 * hidden_size+1] 
	% delta_1 [hidden_size * feature_size + 1]error2 [hidden_size * 1] a_1 [1 * feature_size+1]
	delta_2 = delta_2 + error_3 * a_2;
	delta_1 = delta_1 + error_2 * a_1;	
	
end

% Step 5; Obtain the (unregularized) gradient for the neural network
% cost function by dividing the accumulated gradients by 1/m
Theta2_grad = delta_2 / m;
Theta1_grad = delta_1 / m;



%% Part 3: Regularation part--------------------------------------------------

% J - add regularation item
J = J + lambda * (sum(sum(Theta1(:, 2:end).^2))...
	+ sum(sum(Theta2(:, 2:end).^2))) /(2 * m);

% Gradient - regularation item
Theta1_grad = Theta1_grad + lambda * [zeros(hidden_layer_size, 1) Theta1(:, 2:end)] / m;
Theta2_grad = Theta2_grad + lambda * [zeros(num_labels, 1) Theta2(:, 2:end)] / m;


%% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
