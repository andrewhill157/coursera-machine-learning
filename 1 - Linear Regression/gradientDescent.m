function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
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
    m = length(X);
    
    % Andrew's Notes:
    % Remember! Matrix-matrix multiplication is not commutative. You have
    % to make sure you are ordering the matrices correctly when doing
    % multiplication, no matter what the formula says. 
    
    % Note that (1/m)*X'*(X*theta-y) is simply the derivative of the cost
    % function with respect to theta
    % Multiplying by X' implicitly calculates the sum, by multiplying the
    % two vectors together, so no sum required. 
    
    % See lecture notes for why this formula works out this way (had
    % trouble getting this to work...)
    theta = theta - alpha*(1/m)*X'*(X*theta-y);



    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end
    % Plot used to make sure cost always decreases
    plot(J_history)
end
