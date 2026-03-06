% show command history!!!

net = alexnet;
plot(net)
analyzeNetwork(net)
%%
% Reading images from folder
%cd ./test
%im = imread()
% 
ls
% how to read image into matlab
im = imread('img2.JPG');
% dimensions?
image(im)
imshow(im)
%% Asking the network to classify the image
net.classify(im)
%% lets look on network
net,
% let's look on network Layers
net.Layers
% last layers is related with classifying task
net.Layers(end)
% look which is 
net.Layers(end).Classes
%% Also look on first layer - size of images
net.Layers(1)
%% look on learnable parameters on 2nd layers and other layers
% 
%% How to feed all images into my network
% look on image folder

% 1.st option loop/e.t.c.
    %read images in loop and put that in 4rd dimention
    for i = 1:8
        im(:,:,:,i) = imread(sprintf('img%d.JPG',i));
    end
net.classify(im);
% 2nd option 
imgds = imageDatastore(".");
% path to datastore
im1 = read(imgds);
imshow(im1)
% again
% again
im1 = readimage(imgds,3);
image(im1)
montage(imgds)
montage(imgds,'Size',[2 2])
%% if image size does not fit to neural network input
imgds2 = imageDatastore("test2")
% net.classify(imgds2)
im1 = read(imgds2); 
imres = imresize(im1,[227 227]);
image(imres);
% how to resize a whole datastore
augs = augmentedImageDatastore([227 227],imgds2);
classify(net,augs)
montage(imgds2)
%% Data store with black and white images
imgds3 = imageDatastore(".\test3")
augs = augmentedImageDatastore([227 227],imgds3,"ColorPreprocessing","gray2rgb");
classify(net,augs)
%% Training of neural network 

