

% This demo computes the DEA, PCA, and NNMF components at first and then 
% computes the classification performance
% If you do not want to train the network, please use
% demo_DRwithTrainedModel.m, where we used trained model.
% Please download the data and trained model from Google Drive link
% https://drive.google.com/drive/folders/1xNvf5tHNHW2dQPmEuyrfEPLtACQSiKoY?usp=sharing
%% Load data
load data_DR.mat


irow=224;
icol=224;
A = double(reshape(dataFundus,10538,irow*icol));

%% Divide the dataset into training and testing

perc=0.8;
[TrainData, TestData, TrainLabel, TestLabel]=divideTrainTest(zscore(A),groundTruth,perc);

% Latent dimension
reduced_Dim=32;

% Compute PCA components
[~, ~, pcaComponents,~] = pca_comp(TrainData',reduced_Dim);

% Compute NNMF components
[~, ~, nmfComponents] = nmf_comp(TrainData',reduced_Dim);

% Compute DEA components
maxEPOCH=5000;
[~, ~, deaComponents] = dea_comp(TrainData',reduced_Dim,maxEPOCH);

%% Classification of testing data 
trainingFeatures=TrainData*pcaComponents;
classifier = fitcecoc(trainingFeatures', TrainLabel', ...
    'Learners', 'Linear', 'Coding', 'onevsall', 'ObservationsIn', 'columns');

testFeatures=(TestData)*pcaComponents;

predictedLabelsPCA = predict(classifier, testFeatures', 'ObservationsIn', 'columns');
AccuracyTestPCA=sum(TestLabel==predictedLabelsPCA)/numel(TestLabel);


trainingFeatures=TrainData*deaComponents;
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



