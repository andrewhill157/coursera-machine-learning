function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% Add the bias unit to the input layer
rows = size(X, 1)
X = [ones(rows, 1), X];

% Calculate the activations of the hidden layer using Theta1 and the inputs
z2 = Theta1*X';
a2 = sigmoid(z2);

% a2 needs to be transposed to use same equations as first layer
a2 = a2';

% Add the bias unit to the hidden layer
rows = size(a2, 1);
a2 = [ones(rows, 1), a2];

% Compute the output of the third and final layer
all_predictions = sigmoid(Theta2*a2');

% Find which index (predicted number) has the max probability
[C, p] = max(all_predictions, [], 1);

% return the predicted values for all inputs. p has to be transposed to
% work with the code we have been given.
p = p';








% =========================================================================


end
