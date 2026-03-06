%% File was used to demostrate how to estimate training results

%% open image directory:

directory = "./Linnaeus 5 256X256/";
directory = "D:\dokumenti\HPC\neural_network_seminar2\Linnaeus 5 256X256\";
imds = imageDatastore(directory+"test","IncludeSubfolders",...
    true,'LabelSource','foldernames');
augsTest = augmentedImageDatastore([227 227],imds,'ColorPreprocessing','gray2rgb');

%% get correct class names 
realClass = imds.Labels;
classNames = categories(imds.Labels)
load fine_tuning_example.mat

%[predClass,scores] = net.classify(augsTest); old matlab version sample
YTest = minibatchpredict(net,augsTest);%new matlab versions
predClass = scores2label(YTest,classNames);% new matlab verions
confusionchart(realClass,predClass);
%% varbūtība 
prob = nnz(realClass==predClass)/numel(realClass);

%% how to look on wrong images:
indexes_of_wrong_images = realClass~=predClass;
find(indexes_of_wrong_images)  %looking on wrong image

%% piešķirt N jebko no tā ko atrieza find, piešķirt N un palaist
N = ;
img = readimage(imds,N);
imshow(img)
title(sprintf('this is image from category "%s", \n neural net thinks that this is "%s"',...
    realClass(N),predClass(N)))