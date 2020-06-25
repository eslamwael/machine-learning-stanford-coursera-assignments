function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivaties of the cost w.r.t. each parameter in theta


h = sigmoid( X * theta ); %hypothesis function
reg= ( lambda / m ) ; 

% cost before regularization
J_unreg = ( -1 / m )*( y' *log(h) + ( 1 - y )' * log( 1 .-h) ) ;

%excluding theta(1) in the regularization cost term
J_reg = ( reg / 2 ) .* (theta(2:size(theta))' * theta(2:size(theta))) ;

% Adding the regularization cost term
J = J_unreg +  J_reg ;


grad_log = (1 / m ) * ( ( h - y )' * X )' ;
grad = grad_log + reg .* theta ;
grad(1,1) = grad_log(1,1) ; %excluding theta(1)


% =============================================================

end
