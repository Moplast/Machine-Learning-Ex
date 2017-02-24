function [] = PlotValidationCurve(error_train, error_cv, lambda_vec)

figure;
plot(lambda_vec, error_train, lambda_vec, error_cv);
legend('Train', 'Cross Validation');
xlabel('lambda');
ylabel('Error');

fprintf('lambda\t\tTrain Error\tValidation Error\n');
for i = 1:length(lambda_vec)
	fprintf(' %f\t%f\t%f\n', ...
            lambda_vec(i), error_train(i), error_cv(i));
end

end