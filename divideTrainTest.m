function [trainData testData trainLabel testLabel]=divideTrainTest(data,groundTruth,perc)
[m TotalSamples]=size(data);

groundTruth=grp2idx(groundTruth);

num_class=numel(unique(groundTruth));
[GTu,ia,ic]=unique(groundTruth);
trainData=[];testData=[];trainLabel=[];testLabel=[];
for i=1:num_class
    
ind=find(groundTruth==i);    
indL=round(perc*length(ind));
indx=ind(1:indL);
indy = ind(~ismember(ind,indx));
%indy=intersect(ind,indx);

trainData=[trainData;data(indx,:)];
testData=[testData;data(indy,:)];
trainLabel=[trainLabel; groundTruth(indx)];
testLabel=[testLabel; groundTruth(indy)];
 
end

