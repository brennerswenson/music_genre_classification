% THIS FILE IS BASICALLY THE SAME AS THE OTHER TRAINING FILE, BUT ITERATES
% OVER GRID SEARCH POSSIBILITIES AND SAVES THE RESULTS

train_location = fullfile('fma_small/train');
train_ids = imageDatastore(train_location, 'IncludeSubFolders', true, ...
  'ReadFcn', @matRead, 'FileExtensions', '.mat', 'LabelSource', 'foldernames');
 
val_location = fullfile('fma_small/validation');
val_ids = imageDatastore(val_location, 'IncludeSubFolders', true, ...
  'ReadFcn', @matRead, 'FileExtensions', '.mat', 'LabelSource', 'foldernames');

train_ids = subset(train_ids, datasample(1:length(train_ids.Labels), 1500, 'Replace', false));
val_ids = subset(val_ids, datasample(1:length(val_ids.Labels), 500, 'Replace', false));
 
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
numHiddenUnits_1 = 128;
numHiddenUnits_2 = 64;
poolSize_1 = 2;
poolSize_2 = 2;
poolSize_3 = 2;
poolSize_4 = 4;
poolSize_5 = 4;

hidden_units_1_options = {32, 64, 128, 512, 1024, 2048};
hidden_units_2_options = {32, 64, 128, 512, 1024, 2048};

results = [];

for hu1 = 1:length(hidden_units_1_options)
    for hu2 = 1:length(hidden_units_2_options)
        
        disp('Number of hidden units 1');
        disp(hidden_units_1_options{hu1});
        disp('Number of hidden units 2');
        disp(hidden_units_2_options{hu2});
        
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

            fullyConnectedLayer(hidden_units_1_options{hu1},"Name","fc_1")

            dropoutLayer(0.4,"Name","dropout2")

            fullyConnectedLayer(hidden_units_2_options{hu2},"Name","fc_2")

            dropoutLayer(0.4,"Name","dropout3")

            fullyConnectedLayer(numClasses,"Name","fc_3")
            softmaxLayer("Name","softmax")
            classificationLayer("Name","classoutput", 'Classes', categorical(Classes))];

            options = trainingOptions('adam', ...
                'MaxEpochs', 15, ...
                'ValidationData', val_augimds, ...
                'ValidationFrequency', 20, ...
                'MiniBatchSize', 128, ...
                'LearnRateSchedule', 'piecewise', ...
                'LearnRateDropPeriod', 4, ...
                'LearnRateDropFactor', 0.8, ...
                'InitialLearnRate', 0.01, ...
                'L2Regularization', 0.01, ...
                'Verbose', true, ...
                'Shuffle', 'every-epoch', ...
                'ExecutionEnvironment', 'auto', ... 
                'WorkerLoad', 0.60, ...
                'CheckpointPath', '/Users/brennerswenson/Documents/data_science_msc/neural_computing/nuco_coursework/checkpoints');

            net = trainNetwork(train_augimds, layers, options);

            prediction_probs = predict(net, val_augimds);

            % take max classification probabilities
            [max_probabilities, predicted_labels]=max(prediction_probs, [], 2);

            % convert test labels to numeric
            [unique_labels, ~, numeric_labels] = unique(val_ids.Labels);

            accuracy = sum(predicted_labels == numeric_labels) / numel(numeric_labels) * 100;
            
            disp('Accuracy');
            disp(accuracy);
            
            update_values = [accuracy, hidden_units_1_options{hu1}, hidden_units_2_options{hu2}];
            
            results = [results, update_values];
            
            save(sprintf('results_checkpoint_hu1_%d_hu2_%d', hidden_units_1_options{hu1}, hidden_units_2_options{hu2}), 'results')
                                
            save(sprintf('cnn_trained_hu1_%d_hu2_%d', hidden_units_1_options{hu1}, hidden_units_2_options{hu2}), 'net')
    end
end

save('results', 'results')
 
function data = matRead(filename)
inp = load(filename);
f = fields(inp);
data = inp.(f{1});
end
