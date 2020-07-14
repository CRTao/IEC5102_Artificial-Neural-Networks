function [pred_output] = mlp_predict(image, pmeter)
% mlp only with forwarding to final layer

% final weight 
w1 = pmeter.w1;
w2 = pmeter.w2;

% forwarding to end
[Ly1,Ly2,Ly3,z1,z2] = feedforward(image, w1, w2);
[~, argmax] = max(z2);
% get the highest possiblilty of predict data
pred_output = argmax - 1;

end

function [Ly1,Ly2,Ly3,z1,z2] = feedforward(image, w1, w2)
%compute feedforward step

%Sigmoid
% Layer 1
Ly1 = bias_unit(image,0);
z1 = w1*Ly1';
Ly2 = sigmoid(z1);
% Layer 2
Ly2 = bias_unit(Ly2,1);
z2 = w2*Ly2;
Ly3 = sigmoid(z2);

%{
%ReLu
Ly1 = bias_unit(image, 'col');
z1 = w1*Ly1';
Ly2 = ReLu(z1);
Ly2 = bias_unit(Ly2, 'row');
z2 = w2*Ly2;
Ly3 = ReLu(z2);
%}
end

%% bias function
function [X_new] = bias_unit(image, how)
% add (1) to array at index 1 of col/row

%col
if how == 0
    X_new = ones(size(image,1), size(image,2)+1);
    X_new(:,2:end) = image;
%row
elseif how ==1
    X_new = ones(size(image,1)+1, size(image,2));
    X_new(2:end,:) = image;
end
end

%% activation function
function [sig] = sigmoid(z)
%compute the sigmoid function
sig = 1./(1+exp(-z));
end

function [ReL] = ReLu(z)
%compute the sigmoid function
ReL = max(0,z);
end

