%% Loading Data
X = importdata('words_train.txt');
Y = importdata('genders_train.txt');
X = sparse(X);

%% Loading Libs
addpath('./liblinear');


%% CV to find the max acc for different c
crange = 10.^[-3:0.2:1];
for i = 1:numel(crange)
    acc(i) = train(Y, X, sprintf('-s 6 -v 5 -c %g', crange(i)));
end
[~, bestc] = max(acc);
fprintf('Cross-val chose best C = %g\n', crange(bestc));

%% Getting c = 0.0158 is working the best
model = train(Y, X, '-s 6 -c 0.0158');
X_test = importdata('words_test.txt');

%% Predicting
[plabel, acc, prob] = predict(Y(1:4997,1), sparse(X_test), model, '-b 1');