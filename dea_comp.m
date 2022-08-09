



function [m,A,Eigenfaces]=dea_comp(Training_Data,reduced_Dim,maxEPOCH)
% This function computes the DEA components. 
% Inputs: Training_Data: Data in feature X observation format
% reduced_Dim: dimension of latent space of the autoencoder (3rd optimization) of DEA
% maxEPOCH: Maximum epoch of training of autoenecoder in DEA
sz=size(Training_Data);dim_Original=sz(2);
m = mean(Training_Data,2);
Train_Number = size(Training_Data,2);
temp_m = [];  
for i = 1 : Train_Number
    temp_m = [temp_m m];
end
A = double(Training_Data) - temp_m;

% redKdim is the intermediate dimension after first optimization
if dim_Original>128
    redKdim=128;
else
    redKdim=dim_Original;
end

% First optimization
[~,~,~,distance_Matrix]=grpOPT(A,redKdim,'Distance','correlation');

% Second optimization
idx = kernelOPT(distance_Matrix);
Nidx=length(idx);
NidxS=round(0.8*Nidx);
distance_Matrix=distance_Matrix(:,idx(1:NidxS));

% Third optimization
Eigenfaces=networkOPT(distance_Matrix,reduced_Dim,maxEPOCH);

end
