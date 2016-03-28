%% Loading Files
Xtest = load('words_test.txt');
Xtrain = load('words_train.txt');
Ytrain = load('genders_train.txt');

%% NB Model

% We used matlab biult-in naive bayes training methode, and it worked
% pretty good comparing to those online libs.

mNBModel = fitNaiveBayes(Xtrain,Ytrain,'Distribution','mn');

Ypred = predict(mNBModel,Xtest);