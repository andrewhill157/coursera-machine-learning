function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
% Calculate the activations of the hidden layer
rows = size(X,1); 
X = [ones(rows,1), X];
z2 = Theta1*X';
a2 = sigmoid(z2);
a2 = a2';

% Calculate the activation of the output layer
rows1 = size(a2,1);
a2 = [ones(rows1, 1), a2];
h = sigmoid(Theta2*a2');
h = h';


% Convert the y values to matrix where a the respective column 1-m is set to 1
% Each column will be a vector with a one in the respective row. See
% assignment description. 
ym = zeros(num_labels, m);
for i =1:m;
       ym(y(i),i)=1;
end

% Calculate the value of the cost function
% Note that the Theta values corresponding to the bias units are not
% regularized by convention. 
J = (1/m)*sum(sum(-ym'.*log(h) - (1-ym)'.*log(1-h))) + ... % Unregularized cost function
    lambda/(2*m)*(sum(sum(Theta1(:, 2:end).^2))+ sum(sum(Theta2(:, 2:end).^2))); % Added regularization terms

% Now do backpropagation
% Normally, would not do this twice (already computed to calculate cost)
% But this will make the code easier to understand for learning purposes
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% Perform backprop for every example. Vectorized version is complicated 
% according to the instructor.
for i = 1:m
    % Forward propagation
    a1 = X(i,:); 
    z2 = a1*Theta1'; 
    a2 = sigmoid(z2);
    
    a2 = [1, a2];  % Add the bias unit for this layer
    z3 = a2*Theta2'; 
    a3 = sigmoid(z3); 
    
    % Calculate output layer errors
    error3 = a3 - ym(:, i)' ; % 1*10 
    
    % Calculate input layer errors
    error2 = error3*Theta2(:,2:end).*sigmoidGradient(z2); % 1*25
    
    % Use errors to calculate the cummulative gradients
    Theta1_grad = Theta1_grad + (error2'*a1); % 25x401
    Theta2_grad = Theta2_grad + (error3'*a2); % 10x26
end

% Use gradients to calulate the partial derivative terms needed. 
% Note that the bias terms (first column) are not regularized.
% They are instead replaced by a column of all zeroes.

Theta1_grad = [Theta1_grad/ m] + ... % Unregularized
    [zeros(size(Theta1,1), 1), (lambda/m)*Theta1(:, 2:end)]; % added regularization terms
Theta2_grad = [Theta2_grad/ m] + ... % Unregularizedsu
    [zeros(size(Theta2,1), 1), (lambda/m)*Theta2(:, 2:end)]; % added regularization terms

% Unroll the parameters for grad1 and grad2 into a single vector for 
% minnimization functions which expect this format.
grad = [Theta1_grad(:); Theta2_grad(:)];


% -------------------------------------------------------------

% =========================================================================


end
