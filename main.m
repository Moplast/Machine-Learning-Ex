%% Created by @Moplast
% 2017-02-24 
% Neural Network Training

%% Initialization
clear ; close all; clc

% training network parameters
input_layer_size  = 3;   % input features 
hidden_layer_size = 8;   % hidden units
num_labels = 5;          % output labels

% parameters
lambda = 0.03;			% lambda for regularization
epsilon_init = 0.12;	% epsilon for randomly initialize network weights
max_iter = 50;			% max iteration times

% options
test_option = 1;
normalization_option = 1;
mapping_option = 0;

% input and output columns
input_columns = [1, 2, 5];
output_columns = [3, 4];

% Load Training Data
fprintf('-------------------------------------------------------------\n');
fprintf('>> Loading Data ...\n')
data = load('mothershuju.mat');
data = data.mothershuju;
file_col = 2;

% Separate input and output
X = data(:, input_columns);
Y = data(:, output_columns);

% No. of training example 
m = size(X, 1);

% separate test dataset
if test_option == 1
	[X, Y, ~, ~, ~,...
		X_test, y_test ] = SeparateDataset(X, Y, file_col, 0.8, 0);
	m_test = length(y_test);
end

% % Randomly select 100 data points to display
% sel = randperm(size(X, 1));
% sel = sel(1:100);
% 
% plot(X(sel, 1), X(sel, 2), X(sel, 3));

%% Part 1: Preprocess data

fprintf('-------------------------------------------------------------\n');
fprintf('>> Preprocessing Data ...\n');

[ X, y, num_labels, input_layer_size ] = PreprocessData(X, Y, file_col,...
	mapping_option, normalization_option);

if test_option == 1	
	if mapping_option == 1
		[X_test, input_layer_size] = FeatureMapping(X_test);
	end
	if normalization_option == 1
		[ X_test ] = FeatureNormalize(X_test);
	end
end
%% Part 2: Initial weights and Cost function

fprintf('-------------------------------------------------------------\n');
fprintf('Initializing Neural Network Parameters ...\n');

% Initializing parameters
initial_nn_params = InitNNTheta( input_layer_size, hidden_layer_size, num_labels, ...
	epsilon_init);

% cost function
J = NNCostFunction(initial_nn_params, input_layer_size, hidden_layer_size, ...
                   num_labels, X, y, lambda);
			   
fprintf('# initial cost: %.3f\n', J)

%% Part 3: Training Neural Networks

fprintf('-------------------------------------------------------------\n');
fprintf('>> Training Neural Network... \n')

% Training NN
[nn_params, cost] = TrainingNN(X, y, ...
	input_layer_size, hidden_layer_size, num_labels, ...
	lambda, max_iter, initial_nn_params);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

fprintf('# final cost: %.3f\n', cost(end));
fprintf('-------------------------------------------------------------\n');

%% Part 4: Implement Training example predict

% predict training examples
pred = Predict(Theta1, Theta2, X);

fprintf('Training Set Accuracy: %f\n', mean(double(pred == y)) * 100);

%% Part 5: Implement Test examples predict

if test_option == 1
	% predict test dataset
	pred_test = Predict(Theta1, Theta2, X_test);

	fprintf('\nTest Set Accuracy: %.3f\n', mean(double(pred_test == y_test)) * 100);
end

%% Part 6: Tuning parameters

[X_train, ~, y_train,...
	X_cv, y_cv, ...
	X_test, y_test ] = SeparateDataset(X, Y, file_col, ...
	0.3, 0.4);

[ X_train, ~, ~] = FeatureNormalize(X_train);
[ X_cv, ~, ~] = FeatureNormalize(X_cv);


fprintf('-------------------------------------------------------------\n');
fprintf('>> Prepare for learning curve...\n');

% Calculate learning curve
[error_train, error_cv] = LearningCurve(X_train, y_train, ...
	X_cv, y_cv, ...
	num_labels, ...
	input_layer_size, hidden_layer_size,...
	lambda, max_iter, epsilon_init);

% Plot learning curve
PlotLearningCurve(error_train, error_cv);


% choose lambda
fprintf('-------------------------------------------------------------\n');
fprintf('>> Prepare for validation curve...\n');

[lambda_vec, error_train, error_cv] = ValidationCurve(X_train, y_train, ...
	X_cv, y_cv,...
	num_labels,...
	input_layer_size, hidden_layer_size, ...
	epsilon_init, max_iter);

PlotValidationCurve(error_train, error_cv, lambda_vec);

%% END

fprintf('-------------------------------------------------------------\n');
fprintf('>> Program finished.\n');
