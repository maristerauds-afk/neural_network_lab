%% example for training from the scratch to be send to HPC
%% images was rotated, shifted and mixed for better result
% you have to modify directory
% it will be path to the images
% but download Linnaeus data set first
% and place in neural_network_training_sets folder
% Linnaeus 5 256X256 subfolder
% (link is in neural_network_training_sets folder)
% e.g.
% .\neural_network_training_sets\Linnaeus 5 256X256

%directory = "D:\from_Downloads\Linnaeus 5 256X256\";


%layers = alexnet('Weights','none')
clear
load alexnet_5_classes_untrained.mat
layers = layers_1;

    directory = "../Linnaeus 5 256X256/";
imds = imageDatastore(directory+"train","IncludeSubfolders",...
    true,'LabelSource','foldernames');
[imdsTrain ,imdsValidation] = splitEachLabel(imds,700);
pixelRange = [-32 32];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange);
%augsTrain = augmentedImageDatastore([227 227],imdsTrain,'ColorPreprocessing','gray2rgb',...
 %   'DataAugmentation',imageAugmenter);
augsTrain = augmentedImageDatastore([227 227],imdsTrain,'ColorPreprocessing','gray2rgb');
augsValidation = augmentedImageDatastore([227 227],imdsValidation,'ColorPreprocessing','gray2rgb');

opts = trainingOptions("sgdm");
%opts.Shuffle = 'every-epoch';
opts.ValidationData = augsValidation;
opts.ValidationFrequency = floor(numel(imdsTrain.Files)/128);
opts.Plots = 'training-progress';
opts.InitialLearnRate = 0.001;
opts.MaxEpochs = 20;
[net,info] = trainNetwork(augsTrain,layers,opts);
save full_training_example2.mat

