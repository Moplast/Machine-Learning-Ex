function W = randInitializeWeights(Layer_in, Layer_out, epsilon_init)
%RANDINITIALIZEWEIGHTS Randomly initialize the weights of a layer with L_in
%incoming connections and L_out outgoing connections
%   W = RANDINITIALIZEWEIGHTS(L_in, L_out) randomly initializes the weights 
%   of a layer with L_in incoming connections and L_out outgoing 
%   connections. 

% initialization
W = zeros(Layer_out, 1 + Layer_in);

% randomly pick values between [-epsilon_init, epsilon_init]
W = rand(Layer_out, Layer_in + 1) * 2 * epsilon_init - epsilon_init;

% =========================================================================

end
