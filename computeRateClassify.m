

function rateClassify=computeRateClassify(TestData,mp,Ap,Eigenfaces,TrainLabel,TestLabel)

[m1,n1,TotalTestSamples] = size(TestData);

for i=1:TotalTestSamples
TestImage=TestData(:,:,i);
Recognized_index = recog_code(TestImage, mp, Ap, Eigenfaces);
ID=Recognized_index;
TestResult(i) = TrainLabel(ID);
end
Result = (TestResult == TestLabel');


rateClassify = 100*(sum(Result/TotalTestSamples));