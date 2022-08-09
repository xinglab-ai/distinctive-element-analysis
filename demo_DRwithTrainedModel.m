
% This demo loads the trained models and then 
% computes the classification performance
% Please download the data and trained model from the following Google Drive link
% https://drive.google.com/drive/folders/1xNvf5tHNHW2dQPmEuyrfEPLtACQSiKoY?usp=sharing

%% Load data and trained models
load trainedModel.mat
load data_DR.mat


irow=224;
icol=224;
A = double(reshape(dataFundus,10538,irow*icol));

%% Divide the dataset into training and testing

perc=0.8;
[TrainData, TestData, TrainLabel, TestLabel]=divideTrainTest(zscore(A),groundTruth,perc);


%% Classification of testing data 
trainingFeatures=TrainData*pcaComponents;
classifier = fitcecoc(trainingFeatures', TrainLabel', ...
    'Learners', 'Linear', 'Coding', 'onevsall', 'ObservationsIn', 'columns');

testFeatures=(TestData)*pcaComponents;

predictedLabelsPCA = predict(classifier, testFeatures', 'ObservationsIn', 'columns');
AccuracyTestPCA=sum(TestLabel==predictedLabelsPCA)/numel(TestLabel);


trainingFeatures=TrainData*deaComponents; % From 64 DEA components,
% the components corresponding to the background are removed. Please see
% the manuscript.
classifier = fitcecoc(trainingFeatures', TrainLabel', ...
    'Learners', 'Linear', 'Coding', 'onevsall', 'ObservationsIn', 'columns');

testFeatures=TestData*deaComponents;

predictedLabelsDEA = predict(classifier, testFeatures', 'ObservationsIn', 'columns');
AccuracyTestDEA=sum(TestLabel==predictedLabelsDEA)/numel(TestLabel);




trainingFeatures=(TrainData)*nmfComponents;
classifier = fitcecoc(trainingFeatures', TrainLabel', ...
    'Learners', 'Linear', 'Coding', 'onevsall', 'ObservationsIn', 'columns');

testFeatures=(TestData)*nmfComponents;

predictedLabelsNNMF = predict(classifier, testFeatures', 'ObservationsIn', 'columns');
AccuracyTestNNMF=sum(TestLabel==predictedLabelsNNMF)/numel(TestLabel);



