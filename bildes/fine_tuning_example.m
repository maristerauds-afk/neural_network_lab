%% example for predefined training to be send to HPC with Rotation, Shuffle, Reflection

%% example for pretrained network retraining
% image datastore is in
% Linnaeus 5 256X256 folder (already prepaired for you)
% (link is in neural_network_training_sets folder)
% e.g.
% .\neural_network_training_sets\Linnaeus 5 256X256

%directory = "D:\from_Downloads\Linnaeus 5 256X256\";
%load alexnet_with_weights_modified_for_5_classes.mat
load alexnet_with_weights_modified_for_5_classes.mat
prettrained_net = net_1;
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
opts.Metrics = 'accuracy';

%[net,info] = trainNetwork(augsTrain,prettrained_net,opts);% for old matlab
%versions
[net,info] = trainnet(augsTrain,prettrained_net,'crossentropy',opts);% for new matlab versions
save fine_tuning_example.mat net
