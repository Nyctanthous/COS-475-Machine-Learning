function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

c_guesses = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sigma_guesses = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
prediction_err = zeros(size(c_guesses, 2), size(sigma_guesses, 2));

for i = 1:size(prediction_err, 1)
   for j = 1:size(prediction_err, 2)
       model = svmTrain(X, y, c_guesses(i), @(x1, x2) ...
           gaussianKernel(x1, x2, sigma_guesses(j)));
       predictions = svmPredict(model, Xval);
       prediction_err(i, j) = mean(double(predictions ~= yval));
   end
end

[~, ind] = min(prediction_err(:));
[c_index, sigma_index] = ind2sub(size(prediction_err), ind);

C = c_guesses(c_index);
sigma = sigma_guesses(sigma_index);

end
