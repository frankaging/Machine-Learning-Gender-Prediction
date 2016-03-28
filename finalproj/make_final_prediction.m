
function predictions = make_final_prediction(model,XTest, XTrain)
addpath('./liblinear');
addpath('./libsvm');
% Input
% X_test : a nxp vector representing "n" test samples with p features.
% X_test=[words images image_features] a n-by-35007 vector
% model : struct, what you initialized from init_model.m
%
% Output
% prediction : a nx1 which is your prediction of the test samples

% Sample model

XTrain = XTrain(:,1:5000);
XTest = XTest(:,1:5000);

Ktest = kernel_intersection(XTrain,XTest);

[~, ~, pSVM] = svmpredict(zeros(size(XTest,1),1), [(1:size(Ktest,1))' Ktest], model.SVM, '-b 1');
[~,~, pLi] = predict(zeros(size(XTest,1),1), sparse(XTest), model.Linear, '-b 1');

P = pSVM.*0.7 + pLi.*0.3;

predictions = zeros(size(XTest,1),1);
predictions((P(:,2)>0.5)) = 1;


