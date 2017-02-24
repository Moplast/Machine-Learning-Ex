function [] = PlotLearningCurve(error_train, error_cv)

m = length(error_train);

% plot curve

plot(1: m, error_train, 1 : m, error_cv);
title('Learning curve for Neural Network');
legend('Train', 'Cross Validation');
xlabel('Number of training examples');
ylabel('Error');
% axis([0 13 0 150])

fprintf('# Training Examples\tTrain Error\tCross Validation Error\n');
for i = 1 : m
    fprintf('  \t%d\t\t%f\t%f\n', ...
		i, error_train(i), error_cv(i));
end


end