

function deaComp=networkOPT(dataIn,reduced_Dim,maxEPOCH)
% Encoder-decoder optimization of DEA
% dataIn: input data, reduced_Dim: latent dimension of the encoder
% maxEPOCH: maximum number of epoch for training
autoenc = trainAutoencoder(dataIn','hiddenSize',reduced_Dim,'MaxEpochs',maxEPOCH,'DecoderTransferFunction','satlin',....
    'L2WeightRegularization',0.001,'ShowProgressWindow',true,'SparsityProportion',0.05,'SparsityRegularization',1.6,....
    'UseGPU',false);
finalOut_SCA = encode(autoenc,dataIn');
deaComp = finalOut_SCA';