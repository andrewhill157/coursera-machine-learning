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
%               derivatives of the cost w.r.t. each parameter in theta


% Commenting out the terms for regularization shows the original curve and
% how regularization prevents overfitting. 

% Note that for J, it is convention not to regularize the first theta term.
% sum(theta(2:end).^2) does not include theta(1) in the regularization term
J = 1/m*sum(-y'*log(sigmoid(X*theta))-(1-y)'*log(1-sigmoid(X*theta))) + ...
    lambda/(2*m)*sum(theta(2:end).^2); % added term for regularization

% Again, we will not regularize first theta element, 
% [0; theta(2:end)] puts a zero in place of theta(1) to eliminate it from 
% regularization calculations.
grad = 1/m*X'*(sigmoid(X*theta)-y) + ...
    lambda/m*[0; theta(2:end)]; % Added term for regularization


% =============================================================

end
