% Transfer Learning Using AlexNet

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
augImds = augmentedImageDatastore([227,227],trainingImages, ...
    'DataAugmentation',imageAugmenter);


%%
% This very small data set now contains 55 training images and 20
% validation images. Display some sample images.
numTrainImages = numel(trainingImages.Labels);
% idx = randperm(numTrainImages,16);
% figure
% for i = 1:16
%     subplot(4,4,i)
%     I = readimage(trainingImages,idx(i));
%     imshow(I)
% end

%% Load Pretrained Network

   load('alexnet.mat');
   net=alexnet;
% net=inceptionresnetv2();


%% Transfer Layers to New Network

 layersTransfer = net.Layers(1:end-3);
 numClasses = numel(categories(trainingImages.Labels));
 layers = [
     layersTransfer
     fullyConnectedLayer(numClasses,'WeightLearnRateFactor',1,'BiasLearnRateFactor',2)
     softmaxLayer
     classificationLayer];
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
%    
%[varargout{1:aLayer.NumOutputs+aLayer.PrivateNumStates}] = predict( aLayer, varargin{:} );
%%
% Train the network that consists of the transferred and new layers.

 netTransfer = trainNetwork(augImds,layers,options);

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

