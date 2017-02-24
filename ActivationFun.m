function [ g , gd ] = ActivationFun( z )
% activation function of NN : tanh
% return function value $g$ and gradient $gd$

% % sigmoid(2z)
% s = 1.0 ./ (1.0 + exp(-2 * z));
% 
% % tanh = 2sigmoid(2z)-1
% g = 2 * s - 1;
% 
% % gradient
% gd = 1 - g.^2;

g = 1.0 ./ (1.0 + exp(-z));

gd = g .* (1-g);

end