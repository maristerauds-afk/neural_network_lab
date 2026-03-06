%% Paņems bildes no dotā ceļa (mapju nosaukumus izmantos kā klašu nosaukumus)
digitDatasetPath = fullfile(matlabroot,'toolbox','nnet', ...
    'nndemos','nndatasets','DigitDataset');
imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');
%% Display some images of datastore
figure
numImages = 10000;
perm = randperm(numImages,20);
for i = 1:20
    subplot(4,5,i);
    imshow(imds.Files{perm(i)});
    drawnow;
end
%% Sadalam bildes priekš tīkla trenēšanas un tīkla testēšanas
numTrainingFiles = 750;
[imdsTrain,imdsTest] = splitEachLabel(imds,numTrainingFiles,"randomize");
%% Sadalam bildes priekš tīkla trenēšanas un tīkla testēšanas un validācijas
% lets divide for such parts 70:20:10 Learning-Validation-Testing
% numTrainingFiles = 0.7;
% numValidationFiles = 0.2;
%[imdsTrain,imdsValidation,imdsTest] = ...
%    splitEachLabel(imds,numTrainingFiles,numValidationFiles,"randomize");
%% šeit tiek sagatavots neirona tīkla struktūra (šim mums būs vairāk uzskatāms piemērs)
layers = [...
    imageInputLayer([28 28 1])
    convolution2dLayer(5,20)
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];
%% Liekam trenešanas opcijas:
options = trainingOptions('sgdm');% solver name
%options.ValidationData = imdsValidation;
options.MaxEpochs = 10;% cik reizēs iet cauri visām bildēm
options.MiniBatchSize = 128;% number of images read at once
options.Momentum = 0.9;
% 1 - maximal contribution of previous step
% 0 - minimal contribution of previous step
options.InitialLearnRate = 5e-3;% cik ātri uztrenējas
% options.Plots = 'training-progress';
 options.Verbose = true;
 %options.LearnRateSchedule = 'piecewise';
 %options.LearnRateDropFactor = 0.1;
 %options.LearnRateDropPeriod = 4;
%% visbeidzot trenējam tīklu

net = trainNetwork(imdsTrain,layers,options);
