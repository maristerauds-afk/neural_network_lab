%% example for predefined training to be send to HPC
% you have to modify directory
% it will be path to the images
% but download Linnaeus data set first
% and place in neural_network_training_sets folder
% Linnaeus 5 256X256 subfolder
% (link is in neural_network_training_sets folder)
% e.g.
% .\neural_network_training_sets\Linnaeus 5 256X256

%directory = "D:\from_Downloads\Linnaeus 5 256X256\";
load alexnet_with_weights_modified_for_5_classes.mat
directory = "../Linnaeus 5 256X256/"


imds = imageDatastore(directory+"train","IncludeSubfolders",...
    true,'LabelSource','foldernames');
[imdsTrain ,imdsValidation] = splitEachLabel(imds,700);
pixelRange = [-32 32];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange);
augsTrain = augmentedImageDatastore([227 227],imdsTrain,'ColorPreprocessing','gray2rgb',...
    'DataAugmentation',imageAugmenter);
augsValidation = augmentedImageDatastore([227 227],imdsValidation,'ColorPreprocessing','gray2rgb');


opts = trainingOptions("sgdm");
options.MaxEpochs = 10;% 

opts.InitialLearnRate = 0.001;
%opts.LearnRateDropFactor = 0.5;
%opts.LearnRateDropPeriod = 10;
opts.MiniBatchSize= 128;
%opts.LearnRateSchedule = 'none';
opts.ValidationData = augsValidation;
opts.ExecutionEnvironment = 'gpu';%|'cpu'|'multi-gpu'|'parallel'|
opts.Plots = 'training-progress';
[net,info] = trainNetwork(augsTrain,layers_1,opts);


save example41.mat
