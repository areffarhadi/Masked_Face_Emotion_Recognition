
%%% Download the NATLAB version of VGGFace2 model
%%% (https://drive.google.com/drive/folders/1CCniQEl1uQB94Zg2vmSnzv-cIpFmsazL?usp=sharing)

  close all
  clear

trainingImages = imageDatastore('.\DATASET2\SI\TRAIN',...
"IncludeSubfolders",true,"LabelSource","foldernames");
validationImages = imageDatastore('.\DATASET2\SI\test',...
"IncludeSubfolders",true,"LabelSource","foldernames");




imageAugmenter = imageDataAugmenter( ...
    'RandRotation',[-20,20], ...
    'RandXTranslation',[-5 5], ...
    'RandYTranslation',[-5 5]);
augImds = augmentedImageDatastore([224,224],trainingImages, ...
    'DataAugmentation',imageAugmenter);
%%

numTrainImages = numel(trainingImages.Labels);

%% Load Pretrained Network

 load('VggFace2.mat');

%% Transfer Layers to New Network

lgraph = layerGraph(net);
numClasses = numel(categories(trainingImages.Labels));

newFCLayer =  fullyConnectedLayer(numClasses,'WeightLearnRateFactor',1,'BiasLearnRateFactor',2);
lgraph = replaceLayer(lgraph,'classifier_low_dim',newFCLayer);
newClassificatonLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph,'ClassificationLayer_classifier_low_dim',newClassificatonLayer);
 %%%
%%

miniBatchSize = 16;
numIterationsPerEpoch = floor(numel(trainingImages.Labels)/miniBatchSize);
options = trainingOptions('sgdm',...
    'MiniBatchSize',miniBatchSize,...
    'MaxEpochs',20,...
    'InitialLearnRate',1e-4,...
    'Plots','training-progress',...
    'Verbose',false,...
    'ValidationData',validationImages,...
    'Shuffle', 'every-epoch', ...
    'ValidationFrequency',numIterationsPerEpoch);
    
%%
% Train the network that consists of the transferred and new layers.

netTransfer = trainNetwork(augImds,lgraph,options);

save('netTransfer','netTransfer');
%% Classify Validation Images
% Classify the validation images using the fine-tuned network.
predictedLabels = classify(netTransfer,validationImages);

%%
% Calculate the classification accuracy on the validation set. Accuracy is
% the fraction of labels that the network predicts correctly.
valLabels = validationImages.Labels;
accuracy = mean(predictedLabels == valLabels);

%%
% Calculate the classification accuracy on the validation set. 
classes={'AN';'DI';'FE';'HA';'SA'};

predictedLabels = classify(netTransfer,validationImages);
yp = predict(netTransfer,validationImages);

valLabels = validationImages.Labels;
accuracy = mean(predictedLabels == valLabels);
disp(accuracy);

cm=confusionchart(validationImages.Labels,predictedLabels);
figure()
plotconfusion(validationImages.Labels,predictedLabels);

[C,order] = confusionmat(validationImages.Labels,predictedLabels);
stats = statsOfMeasure(C, 1);

figure()
rocObj = rocmetrics(validationImages.Labels,yp,classes);
plot(rocObj,AverageROCType="micro")

