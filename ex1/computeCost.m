function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

% My Notes:
% note that normally theta is transposed in the formula below. However,
% here theta is already passed in its transposed form, so no need. 

% Note that the hypothesis is theta'X when they are set up correctly, which
% is equal to theta(1)+theta(2)*X(2). See notes for definition of cost
% function. 
J = 1/(2*m)*sum((X*theta-y).^2);


% =========================================================================

end
