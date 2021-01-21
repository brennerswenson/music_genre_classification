% __________________________ SVM TRAINING _________________________ %

Classes = {
    'Electronic', ... 
    'Experimental', ...
    'Folk', ...
    'Hip-Hop',...
    'Instrumental', ...
    'International', ...
    'Pop', ...
    'Rock'
};

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

% Load the CNN to extract feature maps
load mlp_trained_best

% conv layer to extract features from 
layer = 'pool5';

% Compute the activations
features_train = activations(net, train_augimds, layer, 'OutputAs', 'rows');
features_val = activations(net, val_augimds, layer, 'OutputAs', 'rows');

% Assign labels to variables
train_labels = train_ids.Labels;
val_labels = val_ids.Labels;

% Specify the hyperparameters used for the SVM
template = templateSVM('KernelFunction', 'gaussian', ...
                        'KernelScale', 15.836, ...
                        'Standardize', true, ...
                        'BoxConstraint', 1);

% BELOW BLOCK OF CODE IS WHAT WAS USED FOR THE BAYESIAN OPTIMISATION %
%--------------------------------------------------------------------%
% template = templateSVM("KernelFunction", "gaussian", ...
%                        "Standardize", true);

% classificationSVM = fitcecoc(features_train, train_labels, ...
% 'Learners', template, 'Coding', 'onevsone', ...
% 'ClassNames', categorical(Classes), 'OptimizeHyperparameters', ... 
% {'BoxConstraint'}, ... 
% 'HyperparameterOptimizationOptions', struct('Verbose', 2, ... 
% 'UseParallel', true, ...
% 'Kfold', 5, 'Optimizer', 'bayesopt', 'MaxTime', 36000, 'Repartition', true));
%--------------------------------------------------------------------%

% Fit the SVM, indicating to use cross validation
classificationSVM = fitcecoc( ...
features_train, ...
train_labels, ...
'Learners', template, ...
'Coding', 'onevsone', ...
'ClassNames', categorical(Classes), ...
'CrossVal', 'on');

% Pull the best performing model'a accuracy and location
[best_loss, best_ind] = min(kfoldLoss(classificationSVM, 'mode', 'individual'));

% Save the best model to a variable
best_svm_trained = classificationSVM.Trained{best_ind};

% Calculate performace on the validation data
pred_labels = predict(best_svm_trained,features_val);
accuracy = sum(pred_labels == val_labels) / numel(val_labels) * 100;

% Plot confusion matrix
confusionchart(pred_labels,val_labels);

% Save model for future use
save('svm_trained_best','best_svm_trained')


function data = matRead(filename)
inp = load(filename);
f = fields(inp);
data = inp.(f{1});
end