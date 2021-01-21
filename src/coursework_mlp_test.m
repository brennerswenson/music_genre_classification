% __________________________ MLP TRAINING _________________________ %

% load in test data
test_location = fullfile('data/test');
test_ids = imageDatastore(test_location, 'IncludeSubFolders', true, ...
  'ReadFcn', @matRead, 'FileExtensions', '.mat', 'LabelSource', 'foldernames');

% center crop the data to reduce the dimension from 640 to 512
test_augimds = augmentedImageDatastore([128, 512, 1], test_ids, 'OutputSizeMode', 'centercrop');

load mlp_trained_best

% predict on unseen data
prediction_probs = predict(net, test_augimds);

% take max classification probabilities
[max_probabilities, predicted_labels]=max(prediction_probs, [], 2);

% convert test labels to numeric
[unique_labels, ~, numeric_labels] = unique(test_ids.Labels);

% compute performance and create confusion matrix
accuracy = sum(predicted_labels == numeric_labels) / numel(numeric_labels) * 100;
confusionchart(numeric_labels,predicted_labels);

% function to load in files
function data = matRead(filename)
inp = load(filename);
f = fields(inp);
data = inp.(f{1});
end