%% Loading the image
Xtest = load('images_test.txt');
Xtrain = load('images_train.txt');
Ytrain = load('genders_train.txt');

XtestReduced = [];
XtrReduced = [];
%% First, we greyscale, and rescale the image for test set
for i = 1 : 1
    idxtest = reshape(Xtest(i,:),[100,100,3]);
    idxtest = rgb2gray(idxtest);
    idxtest = imresize(idxtest, [50,50]);
    XtestReduced(i,:) = reshape(idxtest,[1,2500,1]);
end

%% For training set
for i = 1 : 1
    idxtr = reshape(Xtrain(i,:),[100,100,3]);
    idxtr = rgb2gray(idxtr);
    idxtr = imresize(idxtr, [50,50]);
    XtrReduced(i,:) = reshape(idxtr,[1,2500,1]);
end

%% However, we preload all the data
load('image.mat');

%% PCA Data

% We are using numpc = 80, because we plot the numps versus the accuracy to
% repreduce the model. It already pass the 90%
[score_train,score_test,numpc] = pca_getpc(train_x,test_x);

%% Using KNN To Fit the Model
mdl = fitcknn(XtrReduced,Ytrain);

%% Prediction
label = predict(mdl,XtestReduced);