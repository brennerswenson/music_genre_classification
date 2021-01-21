% __________________________ SVM TESTING _________________________ %

% load in test data
test_location = fullfile('data/test');
test_ids = imageDatastore(test_location, 'IncludeSubFolders', true, ...
  'ReadFcn', @matRead, 'FileExtensions', '.mat', 'LabelSource', 'foldernames');

% center crop the data to reduce the dimension from 640 to 512
test_augimds = augmentedImageDatastore([128, 512, 1], test_ids, 'OutputSizeMode', 'centercrop');
test_labels = test_ids.Labels;

% load the CNN to extract the feature maps
load mlp_trained_best

% the layer to pull the activations from
layer = 'pool5';

% compute the feature maps from the convolutional layers of the CNN
features_test = activations(net, test_augimds, layer, 'OutputAs', 'rows');

% load the best SVM model
load svm_trained_best

% calculate predictions and accuracy, plotting confusion matrix
pred_labels = predict(best_svm_trained,features_test);
accuracy = sum(pred_labels == test_labels) / numel(test_labels) * 100;
confusionchart(pred_labels,test_labels);

% function to load in files
function data = matRead(filename)
inp = load(filename);
f = fields(inp);
data = inp.(f{1});
end