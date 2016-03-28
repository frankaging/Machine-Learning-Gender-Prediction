
words_train = dlmread('words_train.txt');
words_test = dlmread('words_test.txt');

XTrain = [words_train];
XTest = [words_test];
% Do the same thing for XTest
% copy/paste code snippet here

tic;
model = init_model();
toc;

tic;
predictions = make_final_prediction(model, XTest, XTrain); %where XTest and XTrain are [words images image_features]
toc;