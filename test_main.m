clear all; 
clc;

% Load parameters
load('2training99test100.mat');

% Load data
Data = load('salesmanshuju.mat');
Data = Data.salesmanshuju;

% Preprocess data
file_col = 2;
mapping_option = 0;
normalization_option = 1;

X = Data(:, [1, 2, 5]);
Y = Data(:, [3, 4]);

[ X, y, ~, ~ ] = PreprocessData(X, Y, file_col,...
	mapping_option, normalization_option);


% Predict
pred = Predict(Theta1, Theta2, X);
fprintf('Training Set Accuracy: %f\n', mean(double(pred == y)) * 100);

