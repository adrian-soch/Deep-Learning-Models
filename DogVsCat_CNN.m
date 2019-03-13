%% Based off: https://www.mathworks.com/examples/matlab/community/36291-deep-learning-example-traning-from-scratch-using-cifar-10-dataset
%% Data Source: https://www.kaggle.com/c/dogs-vs-cats

%% Loading Data
categories = {'Dog','Cat'};
rootFolder = 'Train';
augmenter = imageDataAugmenter('RandYReflection', true, ...
     'RandXShear', [0 1], ...
     'RandYShear', [0 1]);

imds = imageDatastore(fullfile(rootFolder, categories),'LabelSource', 'foldernames');
outputSize = [32, 32];
auimds = augmentedImageDatastore(outputSize,imds,'DataAugmentation', augmenter);
%% Defining Model
varSize = 32;
conv1 = convolution2dLayer(5,varSize,'Padding',2,'BiasLearnRateFactor',2);
conv1.Weights = gpuArray(single(randn([5 5 3 varSize])*0.0001));
fc1 = fullyConnectedLayer(64,'BiasLearnRateFactor',2);
fc1.Weights = gpuArray(single(randn([64 576])*0.1));
fc2 = fullyConnectedLayer(2,'BiasLearnRateFactor',2);
fc2.Weights = gpuArray(single(randn([2 64])*0.1));
layers = [
    imageInputLayer([varSize varSize 3]);
    conv1;
    maxPooling2dLayer(3,'Stride',2);
    reluLayer();
    convolution2dLayer(5,64,'Padding',2,'BiasLearnRateFactor',2);
    reluLayer();
    averagePooling2dLayer(3,'Stride',2);
    convolution2dLayer(5,64,'Padding',2,'BiasLearnRateFactor',2);
    reluLayer();
    averagePooling2dLayer(3,'Stride',2);
    fc1;
    reluLayer();
    dropoutLayer(0.5);
    fc2;
    dropoutLayer(0.1);
    softmaxLayer()
    classificationLayer()];
%% Options
options = trainingOptions('adam', ...
    'LearnRateDropFactor', 0.2, ...
    'LearnRateDropPeriod', 8, ...
    'L2Regularization', 0.009, ...
    'MaxEpochs', 110, ...
    'MiniBatchSize', 128, ...
    'Plots', 'training-progress', ...
    'Shuffle','every-epoch', ...
    'Verbose', true);


%% Training
[net, info] = trainNetwork(auimds, layers, options);

%% Load Test Data
rootFolder = 'test69';
imds_test = imageDatastore(fullfile(rootFolder, categories),'LabelSource', 'foldernames');
aimds_test = augmentedImageDatastore(outputSize,imds_test);
%% Spot Testing
labels = classify(net, aimds_test);
n = 4;
im = imread(imds_test.Files{n});
imshow(im);
if labels(n) == imds_test.Labels(n)
   colorText = 'g'; 
else
    colorText = 'r';
end
title(char(labels(n)),'Color',colorText);

%% Full Testing
confMat = confusionmat(imds_test.Labels, labels);
confMat = confMat./sum(confMat,2);
fprintf('Test Accuracy: %f',mean(diag(confMat))*100);
