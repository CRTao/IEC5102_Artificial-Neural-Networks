%% Prepare the dataset
[imgDataTrain, labelsTrain, imgDataTest, labelsTest, img1000Train, labels1000Train, img1000Test, labels1000Test] = prepareData;

%% Prepare the CNN      
%% Parameter Setting
miniBatchSize = 8192;
options = trainingOptions( 'sgdm',...
    'MiniBatchSize', miniBatchSize,...
    'Plots', 'training-progress',...
    'InitialLearnRate', 0.0001);

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

%% Training 1000 samples
net1000 = trainNetwork(img1000Train, labels1000Train, layers, options);
predLabelsTest = net1000.classify(img1000Test);
testAccuracy = sum(predLabelsTest == labels1000Test) / numel(labels1000Test)
figure;
cm = confusionchart(labels1000Test,predLabelsTest);
cm.ColumnSummary = 'column-normalized';
cm.Title = '1000 Sample Confusion Matrix';
analyzeNetwork(net1000)

%% changing graphics and print out samples.
opengl('save', 'software')
warning off images:imshow:magnificationMustBeFitForDockedFigure
perm = randperm(numel(labels1000Test), 150);
actualLabel = grp2idx(labels1000Test(perm));
predictedLabel = grp2idx(net1000.classify(img1000Test(:,:,1,perm)));
subset = [];
for i= 1:150
    img = img1000Test(:,:,1,perm(i));
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
