% __________________________ MLP TRAINING _________________________ %

% Load in training the data
train_location = fullfile('fma_small/train');
train_ids = imageDatastore(train_location, 'IncludeSubFolders', true, ...
  'ReadFcn', @matRead, 'FileExtensions', '.mat', 'LabelSource', 'foldernames');
 

% Load in the validation data
val_location = fullfile('fma_small/validation');
val_ids = imageDatastore(val_location, 'IncludeSubFolders', true, ...
  'ReadFcn', @matRead, 'FileExtensions', '.mat', 'LabelSource', 'foldernames');
 
% Use randomcropping on the training data for data augmentation
train_augimds = augmentedImageDatastore([128, 512, 1], train_ids, 'OutputSizeMode', 'randcrop');
val_augimds = augmentedImageDatastore([128, 512, 1], val_ids, 'OutputSizeMode', 'centercrop');
 
Classes = { ...
    'Electronic', ... 
    'Experimental', ...
    'Folk', ...
    'Hip-Hop', ...
    'Instrumental', ...
    'International', ...
    'Pop', ...
    'Rock' ...
    };

inputSize = [128, 512, 1];
numClasses = 8;
filterSize_1 = 3;
filterSize_2 = 3;
filterSize_3 = 3;
filterSize_4 = 3;
filterSize_5 = 3;
numFilters_1 = 16;
numFilters_2 = 32; 
numFilters_3 = 64;
numFilters_4 = 128;
numFilters_5 = 64;
poolSize_1 = 2;
poolSize_2 = 2;
poolSize_3 = 2;
poolSize_4 = 4;
poolSize_5 = 4;
numHiddenUnits_1 = 128;
numHiddenUnits_2 = 64;

% CNN / MLP construction
layers = [
    imageInputLayer(inputSize,"Name","imageinput");
    
    convolution2dLayer([filterSize_1 filterSize_1], numFilters_1, "Name","conv1","Padding","same", 'Stride', [1 1])
    reluLayer('Name', 'relu1')
    maxPooling2dLayer([poolSize_1 poolSize_1],"Name","pool1","Padding","same", 'Stride', [poolSize_1 poolSize_1])
    batchNormalizationLayer('Name', 'norm1')
    
    convolution2dLayer([filterSize_2 filterSize_2], numFilters_2,"Name","conv2","Padding","same", 'Stride', [1 1])
    reluLayer('Name', 'relu2')
    maxPooling2dLayer([poolSize_2 poolSize_2],"Name","pool2","Padding","same", 'Stride', [poolSize_2 poolSize_2])
    batchNormalizationLayer('Name', 'norm2')
    
    convolution2dLayer([filterSize_3 filterSize_3], numFilters_3,"Name","conv3","Padding","same", 'Stride', [1 1])
    reluLayer('Name', 'relu3')
    maxPooling2dLayer([poolSize_3 poolSize_3],"Name","pool3","Padding","same", 'Stride', [poolSize_3 poolSize_3])
    batchNormalizationLayer('Name', 'norm3')
    
    convolution2dLayer([filterSize_4 filterSize_4], numFilters_4,"Name","conv4","Padding","same", 'Stride', [1 1])
    reluLayer('Name', 'relu4')
    maxPooling2dLayer([poolSize_4 poolSize_4],"Name","pool4","Padding","same", 'Stride', [poolSize_4 poolSize_4])
    batchNormalizationLayer('Name', 'norm4')
    
    convolution2dLayer([filterSize_5 filterSize_5], numFilters_5,"Name","conv5","Padding","same", 'Stride', [1 1])
    reluLayer('Name', 'relu5')
    maxPooling2dLayer([poolSize_5 poolSize_5],"Name","pool5","Padding","same", 'Stride', [poolSize_5 poolSize_5])
    batchNormalizationLayer('Name', 'norm5')
 
    dropoutLayer(0.4,"Name","dropout1")
    
    fullyConnectedLayer(numHiddenUnits_1,"Name","fc_1")
    
    dropoutLayer(0.4,"Name","dropout2")
    
    fullyConnectedLayer(numHiddenUnits_2,"Name","fc_2")
    
    dropoutLayer(0.4,"Name","dropout3")
 
    fullyConnectedLayer(numClasses,"Name","fc_3")
    softmaxLayer("Name","softmax")
    classificationLayer("Name","classoutput", 'Classes', categorical(Classes))];
    
% Majority of hyperparameters
options = trainingOptions('adam', ...
    'MaxEpochs', 30, ...
    'ValidationData', val_augimds, ...
    'ValidationFrequency', 30, ...
    'MiniBatchSize', 128, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropPeriod', 4, ...
    'LearnRateDropFactor', 0.8, ...
    'InitialLearnRate', 0.01, ...
    'L2Regularization', 0.01, ...
    'Verbose', true, ...
    'Plots','training-progress', ...
    'Shuffle', 'every-epoch', ...
    'ExecutionEnvironment', 'auto', ... 
    'WorkerLoad', 0.60, ...
    'CheckpointPath', '/Users/brennerswenson/Documents/data_science_msc/neural_computing/nuco_coursework/checkpoints');
 
net = trainNetwork(train_augimds, layers, options);
 
save('mlp_trained_best','net')
 
function data = matRead(filename)
inp = load(filename);
f = fields(inp);
data = inp.(f{1});
end
