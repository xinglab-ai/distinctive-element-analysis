function [m, A, Eigenfaces] = nmf_comp(Training_Data,reduced_Dim)

%----------------------------Calculate the mean image ------------------------
% ---------------------compute the covariance matrix --------------------------
m = mean(Training_Data,2);
Train_Number = size(Training_Data,2);
temp_m = [];  
for i = 1 : Train_Number
    temp_m = [temp_m m];
end
A = double(Training_Data) - temp_m;


[Eigenfaces,~] = nnmf(A,reduced_Dim);
