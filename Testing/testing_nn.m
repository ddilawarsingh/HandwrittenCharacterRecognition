
%% Initialization
clear ; close all; clc

%% The parameters
input_layer_size  = 64;  % 64x64 Input Images of Characters
hidden_layer_size = 100;   % 100 hidden units
num_labels = 26;          % 26 labels, from A to Z

%% =========== Loading and Visualizing Data =============

% Load Training Data
fprintf('Loading and Visualizing Data ...\n')

load('testingDataSet.mat');
load('displayDataForTesting.mat');
m = size(X, 1);

% Randomly select 100 data points to display
sel = randperm(size(Disp, 1));
sel = sel(1:100);

displayData(Disp(sel, :));

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ================ Loading Pameters ================

fprintf('\nLoading Saved Neural Network Parameters ...\n')

% Load the weights into variables Theta1 and Theta2
load('trainedWeights.mat');

%% ================= Implement Predict =================
%  After training the neural network, we would like to use it to predict
%  the labels.

pred = predict(Theta1, Theta2, X);

fprintf('\nTesting Set Accuracy: %f\n', mean(double(pred == y)) * 100);

fprintf('Program paused. Press enter to continue.\n');
pause;

%  Randomly permute examples
rp = randperm(m);

for i = 1:m
    % Display 
    fprintf('\nDisplaying Example Image\n');
    displayData(Disp(rp(i), :));

    pred = predict(Theta1, Theta2, X(rp(i),:));
    fprintf('\nNeural Network Prediction: %d (Alphabet %c)\n', pred, char(64+pred));
    
    % Pause with quit option
    s = input('Paused - press enter to continue, q to exit:','s');
    if s == 'q'
      break
    end
end

