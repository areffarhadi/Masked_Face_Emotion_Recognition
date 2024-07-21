
close all

%% Load Pretrained Network

load('netTransfer.mat');
%    
%%
% Train the network that consists of the transferred and new layers.
for i=21
    %1:size(validationImages.Files)
  a=imread(validationImages.Files{i,1});
    predictedLabels = classify(netTransfer,a);
    scoreMap = imageLIME(netTransfer,a,predictedLabels);
figure
imshow(a)
hold on
imagesc(scoreMap,'AlphaData',0.5)
colormap jet


[scoreMap,featureMap,featureImportance]  = imageLIME(netTransfer,a,predictedLabels,'Segmentation','grid','NumFeatures',64,'NumSamples',3072);
figure
imshow(a)
hold on
imagesc(scoreMap,'AlphaData',0.5)
colormap jet
colorbar

numTopFeatures = 5;
[~,idx] = maxk(featureImportance,numTopFeatures);
mask = ismember(featureMap,idx);
maskedImg = uint8(mask).*a;
figure
imshow(maskedImg);

figure ()
imshow(a)
end

