%% Prepare the dataset
[imgDataTrain, labelsTrain, imgDataTest, labelsTest, img1000Train, labels1000Train, img1000Test, labels1000Test] = prepareData;

%% Prepare the CNN      
%% Parameter Setting
miniBatchSize = 1024;
options = trainingOptions( 'sgdm',...
    'MiniBatchSize', miniBatchSize,...
    'Plots', 'training-progress',...
    'InitialLearnRate', 0.0001,...
    'MaxEpochs', 10);

layers = [
    imageInputLayer([28 28 1])
    convolution2dLayer(3,16,'Padding',1)
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(3,32,'Padding',1)
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(3,64,'Padding',1)
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];

%% Training all samples
net = trainNetwork(imgDataTrain, labelsTrain, layers, options);
predLabelsTest = net.classify(imgDataTest);
testAccuracy = sum(predLabelsTest == labelsTest) / numel(labelsTest)
figure;
ck = confusionchart(labelsTest,predLabelsTest);
ck.ColumnSummary = 'column-normalized';
ck.Title = 'All Sample Confusion Matrix';
analyzeNetwork(net)

%% changing graphics and print out samples.
opengl('save', 'software')
warning off images:imshow:magnificationMustBeFitForDockedFigure
perm = randperm(numel(labelsTest), 150);
actualLabel = grp2idx(labelsTest(perm));
predictedLabel = grp2idx(net.classify(imgDataTest(:,:,1,perm)));
subset = [];
for i= 1:150
    img = imgDataTest(:,:,1,perm(i));
    At = insertText(img,[0 28],actualLabel(i)-1,'AnchorPoint','LeftBottom','FontSize',8,'BoxColor','yellow');
    if(actualLabel(i)==predictedLabel(i))
        Pt = insertText(At,[18 28],predictedLabel(i)-1,'AnchorPoint','LeftBottom','FontSize',8,'BoxColor','green');    
    else
        Pt = insertText(At,[18 28],predictedLabel(i)-1,'AnchorPoint','LeftBottom','FontSize',8,'BoxColor','red');  
    end
    subset = cat(4,subset,Pt);
end
figure;
montage(subset,'Size', [10 15])
title(['150 Sample Test'])
