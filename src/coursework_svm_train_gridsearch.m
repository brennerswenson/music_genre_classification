% THIS FILE IS BASICALLY THE SAME AS THE OTHER TRAINING FILE, BUT ITERATES
% OVER GRID SEARCH POSSIBILITIES AND SAVES THE RESULTS

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


load mlp_trained_best

% conv layer to extract features from 
layer = 'pool5';

features_train = activations(net, train_augimds, layer, 'OutputAs', 'rows');
features_val = activations(net, val_augimds, layer, 'OutputAs', 'rows');

train_labels = train_ids.Labels;
val_labels = val_ids.Labels;

box_constrain_options = {0.1, 1, 3, 5, 10, 50, 100};
kernel_scale_options = {0.1, 1, 3, 5, 10, 50, 100};

results = [];

for box_idx = 1:length(box_constrain_options)
    for ks_idx = 1:length(kernel_scale_options)
        
        disp('Box Constraint');
        disp(box_constrain_options{box_idx});
        disp('Kernel Scale');
        disp(kernel_scale_options{ks_idx});

        template = templateSVM("KernelFunction", "gaussian", ...
                               "BoxConstraint", box_constrain_options{box_idx}, ...
                               "KernelScale", kernel_scale_options{ks_idx}, ...
                               "Standardize", true);

        classificationSVM = fitcecoc( ...
        features_train, ...
        train_labels, ...
        'Learners', template, ...
        'Coding', 'onevsone', ...
        'ClassNames', categorical(Classes), ...
        'CrossVal', 'on');

        [best_loss, best_ind] = min(kfoldLoss(classificationSVM, 'mode', 'individual'));

        best_svm_trained = classificationSVM.Trained{best_ind};

        pred_labels = predict(best_svm_trained,features_val);
        accuracy = sum(pred_labels == val_labels) / numel(val_labels) * 100;
        confusionchart(pred_labels,val_labels);
        
        disp('Accuracy');
        disp(accuracy);

        update_values = [accuracy, box_constrain_options{box_idx}, kernel_scale_options{ks_idx}];

        results = [results, update_values];

        save(sprintf('results_checkpoint_box_%d_ks_%d', box_constrain_options{box_idx}, kernel_scale_options{ks_idx}), 'results')

        save(sprintf('svm_trained_box_%d_ks_%d', box_constrain_options{box_idx}, kernel_scale_options{ks_idx}), 'best_svm_trained')
        
    end
end

save('svm_results', 'results')

function data = matRead(filename)
inp = load(filename);
f = fields(inp);
data = inp.(f{1});
end