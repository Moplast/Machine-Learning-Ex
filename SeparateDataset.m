function [X_train, Y_train, y_train,...
	X_cv, y_cv, ...
	X_test, y_test] = SeparateDataset(X, Y, ...
	col, ...
	train_rate, cv_rate)

% Initialization these as different numbers
num_labels_train = 0; num_labels_cv = 1; num_labels_test = 2;

% calculate corresponding dataset size
m = size(X, 1);
m_train = floor(train_rate * m);
m_cv = floor(cv_rate * m);

% train dataset class should has the same size as c validation dataset
while (num_labels_train~=num_labels_cv || ...
	num_labels_test~=num_labels_train || ...
	num_labels_test~=num_labels_cv)

	% rearrange X
	randomNo = randperm(m);

	X_train = X(randomNo(1:m_train), :);
	Y_train = Y(randomNo(1:m_train), :);

	X_cv = X(randomNo(m_train+1: m_train+m_cv), :);
	Y_cv = Y(randomNo(m_train+1: m_train+m_cv), :);

	X_test = X(randomNo(m_train+m_cv+1 : end), :);
	Y_test = Y(randomNo(m_train+m_cv+1 : end), :);

	% Preprocess data
	[ y_train, y_trainMap ] = Numerial2Class(Y_train(:, col)); 
	[ y_test, y_testMap ] = Numerial2Class(Y_test(:, col));
	[ y_cv, y_cvMap ] = Numerial2Class(Y_cv(:, col)); 
	
	num_labels_train = length(y_trainMap);		
	num_labels_test = length(y_testMap);
	if 1-train_rate-cv_rate == 0		
		num_labels_test = num_labels_train;
	end
	num_labels_cv = length(y_cvMap);
	if cv_rate == 0
		num_labels_cv = num_labels_train;
	end

end

end