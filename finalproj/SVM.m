%% Loading Files
XTest = load('words_test.txt');
XTrain = load('words_train.txt');
YTrain = load('genders_train.txt');

%% Add Libs
addpath('./libsvm');

%% We are using intersection kernel SVM
Ktest = kernel_intersection(XTrain,XTest);

%% First we used CV to get the best n values, and then we train the model
model.SVM = svmtrain(Y, [(1:size(K,1))' K], '-s 1 -t 4 -n 0.3 -b 1');

%% Predictions
[~, ~, pSVM] = svmpredict(zeros(size(XTest,1),1), [(1:size(Ktest,1))' Ktest], model.SVM, '-b 1');
P = pSVM;
predictions = zeros(size(XTest,1),1);
predictions((P(:,2)>0.5)) = 1;