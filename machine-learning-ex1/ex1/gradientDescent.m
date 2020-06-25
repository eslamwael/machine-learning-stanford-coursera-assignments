function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
	
	
	%  vectorized solution ( valid for multiple features ) - Single line of code
	h = X*theta ;
	theta = theta - (alpha / m ) * ( ( h - y )' * X )' ;
	
		% Non vectorized solution ( hard for multiple features )
	% h = theta(1) +(theta(2) * X(:,2) );
	%theta_o = theta(1) - ( alpha / m ) * sum(h - y) ;
	%theta_1 = theta(2) - ( alpha / m ) * sum((h - y) .* X(:,2)) ;
	%theta= [ theta_o ; theta_1 ] ;
	
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
	disp(J_history(iter));

end

end
