%% pj2(b) MLP on real data_0516320

%% initialization
rng('default');
clear all;
close all;

%% load data
% classes = 0:9;
% nbr_of_train_samples = 60000;
% nbr_of_test_samples  = 10000;
[img_train, lbs_train, img_test, lbs_test] = loadMNIST();

%% only 1000 of samples
div_train = randperm(60000,1000);
div_test = randperm(10000,1000);
img_train = img_train(div_train,:);
lbs_train = lbs_train(div_train,:);
img_test = img_test(div_test,:);
lbs_test = lbs_test(div_test,:);

%% print training samples
idx = 1:100;
figure;
title('Train Pattern');
displayMNIST(img_train(idx,:));
%figure;
%title('Test Pattern');
%displayMNIST(img_test(idx,:));

%% set parameters
pmeter.n_input = size(img_train,2);
pmeter.n_hidden = 50;
pmeter.n_output = 10;
pmeter.iterations = 15;
pmeter.LearningRate = 0.001;
pmeter.Momentum = 0.5;
pmeter.batchs = 50;

%% multilayer perceptron
[pmeter] = mlp_network(img_train, lbs_train, pmeter);
od_train  = mlp_predict(img_train, pmeter);
od_test  = mlp_predict(img_test, pmeter);

%% accuracy
acc = sum(lbs_train' == od_train) / size(img_train,1);
fprintf('Train Accuracy: %.4f  Correct Samples: %5d Total Samples: %5d\n',(acc * 100),sum(lbs_train' == od_train),size(img_train,1));
acc = sum(lbs_test' == od_test) / size(img_test,1);
fprintf('Test  Accuracy: %.4f  Correct Samples: %5d Total Samples: %5d\n',(acc * 100),sum(lbs_test' == od_test) , size(img_test,1));

%% generate plots
displayFinalMNIST(lbs_test',od_test,img_test(idx,:));

figure;
plot(pmeter.cost/100); 
title({'Training Error';['Test Accuracy:' num2str(acc * 100)]}); grid on;
ylabel('Loss'); xlabel('Iteration');

