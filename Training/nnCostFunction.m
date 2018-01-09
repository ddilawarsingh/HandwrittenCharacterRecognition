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
         
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));


% PART ONE %

X = [ones(m,1),X];  %added bias unit in input layer
Y_matrix = zeros(m,num_labels); 

hx = zeros(m,num_labels);

for i = 1:m
	z2 = Theta1 * (X(i,:))';
	a2 = sigmoid(z2);
	a2 = [1;a2];
	z3 = Theta2 * a2;
	hx(i,:) = (sigmoid(z3))';
	Y_matrix(i,y(i))=1;
end

%temp = 0;
%for i = 1:m
%	for j = 1:num_labels
%		temp = temp + (-Y_matrix(i,j)*log(hx(i,j)))-(1-Y_matrix(i,j))*log(1-hx(i,j));
%	end
%end
%J = temp/m;

J = (sum(sum((((-1*Y_matrix).*log(hx)).-((1-Y_matrix).*log(1-hx))),2)))/m; % vectorized form

temp = 0;
for i = 1:hidden_layer_size
	for j = 2:(input_layer_size+1)
		temp = temp + Theta1(i,j)^2;
	end
end

for i = 1:num_labels
	for j = 2:(hidden_layer_size+1)
		temp = temp + Theta2(i,j)^2;
	end
end

J = J + (lambda/(2*m))*temp;

% PART ONE OVER %

Delta2 = zeros(num_labels,hidden_layer_size+1);
Delta1 = zeros(hidden_layer_size,input_layer_size+1);

for t = 1:m
	a1 = X(t,:)';
	z2 = Theta1 * a1;
	a2 = sigmoid(z2);
	a2 = [1;a2];
	z3 = Theta2 * a2;
	a3 = sigmoid(z3);
	
	d3 = a3 - (Y_matrix(t,:)');
	d2 = (Theta2(:,2:end)' * d3) .* sigmoidGradient(z2);
	
	Delta2 = Delta2 + d3 * a2'; % 10X26
	%Delta2 = Delta2(:,2:end); 
	Delta1 = Delta1 + d2 * a1'; % 25X401
	%Delta1 = Delta1(:,2:end);  
end

for i = 1:hidden_layer_size
	for j = 1:(input_layer_size+1)
		if (j==1)
			Theta1_grad(i,j) = Delta1(i,j)/m;
		else
			Theta1_grad(i,j) = Delta1(i,j)/m + (lambda/m)*Theta1(i,j);
		end
	end
end


for i = 1:num_labels
	for j = 1:(hidden_layer_size+1)
		if (j==1)
			Theta2_grad(i,j)= Delta2(i,j)/m;
		else
			Theta2_grad(i,j) = Delta2(i,j)/m + (lambda/m)*Theta2(i,j);
		end
	end
end

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
