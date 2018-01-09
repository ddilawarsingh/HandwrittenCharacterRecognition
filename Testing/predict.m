function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

m = size(X, 1);
num_labels = size(Theta2, 1);

p = zeros(size(X, 1), 1);

X=[ones(m,1),X];

for i = 1:m
	z2=Theta1*(X(i,:))';
	a2 = sigmoid(z2);
	a2=[1;a2];
	z3=Theta2*a2;
	a3=sigmoid(z3);
	[tempa,p(i)]=max(a3);
end








% =========================================================================


end
