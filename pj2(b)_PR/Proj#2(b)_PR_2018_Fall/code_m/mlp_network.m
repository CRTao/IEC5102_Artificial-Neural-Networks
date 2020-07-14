function [pmeter,Ly3,cost] = mlp_network(image, label, pmeter)
%making network and learn weights from training data
% Input-----------------------------------------------
%
% image : training image data
% label : training label data
% pmeter: parameter of network setting
%
% Output----------------------------------------------
%
% pmeter: Update every cost vs iteration
% Ly3    : Final layer of forward
% cost  : Final cost
%
%----------------------------------------------------

%% parameter
Momentum = pmeter.Momentum;
LearningRate = pmeter.LearningRate;
iterations = pmeter.iterations;
n_output = pmeter.n_output;
batchs = pmeter.batchs;  
pmeter.cost = Inf;


%% initial weight
[w1, w2] = initialize_weights(pmeter);
delta_w1_prev = zeros(size(w1));
delta_w2_prev = zeros(size(w2));

%% Label Categoricalize to onehot
lbs_1hot = onehot_encoding(label,n_output);

%% Training

for i=1:iterations
        
    %minibatch
    nbrPerBatch = floor(length(label)/batchs);
    image = image(1:nbrPerBatch*batchs,:); 
    label = label(1:nbrPerBatch*batchs);
    batch_Input = reshape(1:length(label),batchs,[]);
                  
    for j=1:batchs
        
        idx = batch_Input(j,:);
        % feed forward
        [Ly1,Ly2,Ly3,z1,z2] = feedforward(image(idx,:), w1, w2);
        % compute cost 
        cost = Cal_cost(lbs_1hot(:,idx), Ly3);   
        pmeter.cost = [pmeter.cost, cost];   
        fprintf('Iteration : %d / %d \n',(i-1)*batchs+j,batchs*iterations);
        % compute gradient
        [grad1, grad2] = get_gradient(Ly1,Ly2,Ly3,z1,lbs_1hot(:,idx),w2);
        % update new delta
        delta_w1 = LearningRate * grad1;
        delta_w2 = LearningRate * grad2;
        w1 = w1 - (delta_w1 + (Momentum * delta_w1_prev));
        w2 = w2 - (delta_w2 + (Momentum * delta_w2_prev));        
        delta_w1_prev = delta_w1;
        delta_w2_prev = delta_w2;
        
    end
  
    
end

pmeter.w1 = w1;
pmeter.w2 = w2;

end

%% Side function
%% Forwarding
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

%% Initialization weight
function [w1, w2] = initialize_weights(pmeter)

n_hidden = pmeter.n_hidden;
n_input = pmeter.n_input;
n_output = pmeter.n_output;

% Randomly build weight to [ nbrPerHiddenLayer * 784+1 ]

w1 = 2*rand(1,n_hidden*(n_input+1))-1; 
w1 = reshape(w1,[n_hidden, n_input+1]);
w2 = 2*rand(1,n_output*(n_hidden+1))-1;
w2 = reshape(w2,[n_output, n_hidden+1]);

end

%% Onehot coding
function [onehot] = onehot_encoding(label, k)
onehot = zeros(k,length(label));
for i = 1:length(label)
    onehot(label(i)+1,i) = 1;
end
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

%% backpropagation gradient
function [grad1, grad2] = get_gradient(Ly1,Ly2,Ly3,z1,lbs_1hot,w2)

%Sigmoid
sigma3 = Ly3 - lbs_1hot;
z1 = bias_unit(z1,1);
sigma2 = w2'*sigma3.*sigmoid_gradient(z1);
sigma2 = sigma2(2:end,:);
grad1 = sigma2*Ly1;
grad2 = sigma3*Ly2';

%ReLu
%{
sigma3 = Ly3 - lbs_1hot;
z1 = bias_unit(z1, 'row');
sigma2 = w2'*sigma3.*ReLu_gradient(z1);
sigma2 = sigma2(2:end,:);
grad1 = sigma2*Ly1;
grad2 = sigma3*Ly2';
%}
end

%% Cost function
function [cost] = Cal_cost(lbs_1hot, output)
term1 = -lbs_1hot.*log(output);
term2 = (1-lbs_1hot).*log(1-output);
cost = sum(term1-term2);
cost = sum(cost);
end

%% activation function
function [sig] = sigmoid(z)
%compute the sigmoid function
sig = 1./(1+exp(-z));
end

function [sg] = sigmoid_gradient(z)
%compute sigmoid gradient
sig = sigmoid(z);
sg = sig.*(1-sig);
end

function [ReL] = ReLu(z)
%compute the sigmoid function
ReL = max(0,z);
end

function [Re] = ReLu_gradient(z)
%compute sigmoid gradient
    ReL = ReLu(z);
    if ReL>=0
        Re=1;
    else
        Re=0;
    end
end