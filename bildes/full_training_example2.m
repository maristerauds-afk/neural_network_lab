%% example for training from the scratch to be send to HPC
%% images was rotated, shifted and mixed for better result
% image directory is 
% Linnaeus 5 256X256 and subfolders


%layers = alexnet('Weights','none')
clear
load alexnet_5_classes_untrained.mat
%layers = layers_1;%old matlab
untrained_net = net_1;

    directory = "../Linnaeus 5 256X256/";
imds = imageDatastore(directory+"train","IncludeSubfolders",...
    true,'LabelSource','foldernames');
[imdsTrain ,imdsValidation] = splitEachLabel(imds,700);
pixelRange = [-32 32];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange,...
    'RandRotation',[-45 45]);
%augsTrain = augmentedImageDatastore([227 227],imdsTrain,'ColorPreprocessing','gray2rgb',...
 %   'DataAugmentation',imageAugmenter);
augsTrain = augmentedImageDatastore([227 227],imdsTrain,'ColorPreprocessing','gray2rgb');
augsValidation = augmentedImageDatastore([227 227],imdsValidation,'ColorPreprocessing','gray2rgb');

opts = trainingOptions("sgdm");
opts.Shuffle = 'every-epoch';
opts.ValidationData = augsValidation;
opts.ValidationFrequency = floor(numel(imdsTrain.Files)/128);
opts.Plots = 'training-progress';
opts.InitialLearnRate = 0.001;
opts.MaxEpochs = 200;
opts.Metrics = 'accuracy';
%[net,info] = trainNetwork(augsTrain,layers,opts);%old matlab
[net,info] =  trainnet(augsTrain,untrained_net,'crossentropy',opts);

save full_training_example2.mat net

