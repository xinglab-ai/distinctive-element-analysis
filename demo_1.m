
%% Demo code for manuscript "Discovering distinctive elements of biomedical and
% biometric data for neuro-inspired interpretable exploration"

clc
clear
load ORL_FaceDataSet;  % Loading face dataset. ORL consists of 40 classes, each comprising 10 samples
A=double(ORL_FaceDataSet);
trng=rng;
load('t1.mat'); % For reproducibility
rng(trng);
%  Specifying the numbers of training and testing samples, and also the
%  number of eigenvectors (DIM)
%-----------------------------------------------------------------------
Num_Class=40;
No_SampleClass=10;

% Reduced number of data components
reduced_Dim=40;

% 90% of the data is used for training and 10% for testing
No_TrainSamples=9;
No_TestSamples=10-No_TrainSamples;

% Separating the dataset into training and testing sets, and then labeling.
%-------------------------------------------------------------------------------------------

[TrainData, TestData]=Train_Test(A,No_SampleClass,No_TrainSamples,No_TestSamples);
[m,n,TotalTrainSamples] = size(TrainData);
[m1,n1,TotalTestSamples] = size(TestData);
[TrainLabel,TestLabel]=LebelSamples(Num_Class, No_TrainSamples, No_TestSamples);

%%
Training_Data = [];
sz=size(TrainData);
for imidx = 1:sz(3)
    
    img = squeeze(TrainData(:,:,imidx));
    [irow icol] = size(img);
    temp = reshape(img',irow*icol,1);   % Reshaping 2D images to 1D image vectors
    Training_Data = [Training_Data temp];
end


% PCA
[mp, Ap, EigenfacesPCA,EigVect] = EigenfaceCore_TH2(Training_Data,reduced_Dim);

% DEA
maxEPOCH=20000;
[mDEA, ADEA, EigenfacesDEA] = dea(Training_Data,reduced_Dim,maxEPOCH);

% NNMF
[mNMF, ANMF, EigenNMF] = nmf_reduction(Training_Data,reduced_Dim);


%% Classification of test face images
%----------------------------------------------------

rateClassify=computeRateClassify(TestData,mp,Ap,EigenfacesPCA,TrainLabel,TestLabel);
CorrectRatePCA =rateClassify

rateClassify=computeRateClassify(TestData,mNMF,ANMF,EigenNMF,TrainLabel,TestLabel);
CorrectRateNNMF =rateClassify

rateClassify=computeRateClassify(TestData,mDEA,ADEA,EigenfacesDEA,TrainLabel,TestLabel);
CorrectRateDEA =rateClassify


