digitDatasetPath ='D:\Tao_Documents\ThisSemesterHW\類神經網路\pj2(b)_PR\Proj#2(b)_凌胤濤_PR_2018_Fall\code_m\mnist_png\training';
imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');
figure;
perm = randperm(60000,36);
for i = 1:36
    subplot(6,6,i);
    imshow(imds.Files{perm(i)});
end
labelCount = countEachLabel(imds)
img = readimage(imds,1);
size(img)
numTrainFiles = 1000;
[imdsTrain,imdsValidation] = splitEachLabel(imds,numTrainFiles,'randomize');
layers = [
    imageInputLayer([28 28 1])
    convolution2dLayer(3,8,'Padding',1)
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(3,16,'Padding',1)
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(3,32,'Padding',1)
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];
options = trainingOptions('sgdm', ...
    'MaxEpochs',4, ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');
net = trainNetwork(imdsTrain,layers,options);
YPred = classify(net,imdsValidation);
YValidation = imdsValidation.Labels;
accuracy = sum(YPred == YValidation)/numel(YValidation)