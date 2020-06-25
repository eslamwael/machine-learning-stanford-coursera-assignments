function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
values = [ 0.01 ; 0.03 ; 0.1 ; 0.3 ; 1 ; 3 ; 10 ; 30 ] ;

val_size = size(values) ;



% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

%{
% intialize error to maximum 
error_init = inf; % infinity 

for i=1:val_size
  for j=1:val_size
    
    model= svmTrain(X, y, values(i), @(x1, x2) ...
    gaussianKernel(x1, x2, values(j))); 
    
    predictions = svmPredict(model, Xval);
    error = mean(double(predictions ~= yval)) ;
    
    if ( error <= error_init )
      C = values(i) ;
      sigma = values(j) ;
      error_init =  error;  
      endif
    

  end

end

fprintf('[C, sigma] = [%f %f]\n', C, sigma);
% =========================================================================

end
%} 

%% My solution ( Another solution )


for i=1:val_size
  for j=1:val_size
    
    model= svmTrain(X, y, values(i), @(x1, x2) ...
    gaussianKernel(x1, x2, values(j))); 
    
    predictions = svmPredict(model, Xval);
    
    % putting the error in 8 x 8 matrix   
    error ( i , j ) = mean(double(predictions ~= yval)) ;
  end

end

%Find the C , sigma corresponding the minimum error

[minval, row] = min(min(error,[],2)); % row of the min error
[minval, col] = min(min(error,[],1)); % coloumn of the min error

C = values(row);
sigma = values(col);

fprintf('[C, sigma] = [%f %f]\n', C, sigma);
% =========================================================================

end
 
