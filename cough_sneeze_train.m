function cough_sneeze_train

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% This code was developed as a part of the "A vagal-brainstem interoceptive
% circuit for cough-like defensive behaviors in mice" (Gannot et al.) by
% Tomer Stern, University of Michigan, Ann Arbor, email: tomers@umich.edu
% The purpose of the code is to load a set of respiratory behaviors of a
% mouse, plot a hierarchical clustering tree to help the researcher identify
% with minimum bias the signature waveform of a cough and a sneeze.
% It then trains a linear SVM to allow automated classification of new sequences.
% It needs no input arguments, but requires some parameter tuning right in
% the first lines.

% MIT License:

% Copyright (c) 2023 Tomer Stern, University of Michigan, Ann Arbor, tomers@umich.edu

% Permission is hereby granted, free of charge, to any person obtaining a
% copy of this software and associated documentation files (the “Software”),
% to deal in the Software without restriction, including without limitation
% the rights to use, copy, modify, merge, publish, distribute, sublicense,
% and/or sell copies of the Software, and to permit persons to whom the
% Software is furnished to do so, subject to the following conditions:
% The above copyright notice and this permission notice shall be included
% in all copies or substantial portions of the Software.
% THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS
% OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
% FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
% THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
% LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
% FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
% DEALINGS IN THE SOFTWARE.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

close all;
clc;

% script parameters:
similarity_threshold = 6; % run the code once to plot the hierarchical clustering tree, then decide on the similarity level (Y axis value)
% that appears to separate the tree such that the formed branches show significantly higher sequence similarity
% within the branches than between. Lastly, change the value of this parameter accordingly and run the code again.

subsequence_to_take = 406:561; % run the code once to plot the "S.D." figure showing the standard deviations of all input sequences. Then,
% identify the range of values around time point 500 that appear more conserved/interesting, indicating the
% relevant respiratory activity. Lastly, change this parameter accordingly and run the code again.

cough_cluster_branches = [300 301 303 304 305]; % run the code once - after setting 'similarity_threshold' and 'subsequence_to_take', look at
% the panel with all the sequences from each branch, and identify the ones that behave statistically
% as we expect from a cough. Lastly, write the cluster numbers of all these clusters here and run code again.

smoothing_param = 0.05; % for smoothing the signals. for no smoothing set value to 0.
zscore_sequences = 1; % whether to normalize amplitude height using Z-score or not - used only for the hierarchical clustering and the linear SVM.
n_peaks_to_take_from_side = 8; % number of local minima and maxima peaks to take around time point 500. Should not change!
xlsx_sheet_names = {'Info', 'Traces'}; % the names of the excel sheets in the input filename
use_normalized_sequences_for_training_classifier = 0; % whether to use Z-score normalization on the sequences for when we construct the decision tree.

% asking the user to identify the excel file with all the training examples:
[file,path] = uigetfile('*.xlsx', 'Please select the *.xlsx training file:');
filename = fullfile(path, file);
clear file path;

% loading data:
[all_sequences, info] = load_and_prepare_data(filename, smoothing_param, xlsx_sheet_names);

% extracting subsequences:
[all_sequences, all_full_sequences] = extract_subsequences(all_sequences, subsequence_to_take);

% Z-score normalizing all sequences and subsequences:
all_full_sequences_non_normalized = all_full_sequences;
[all_sequences, all_full_sequences] = normalize_sequences(zscore_sequences, all_sequences, all_full_sequences); %#ok<*ASGLU>

% showing the clusters and assigning a label to each sequence according to user defined 'cough_cluster_branches':
info = show_hierarchical_clusters(all_sequences, info, similarity_threshold, cough_cluster_branches);

% showing similarity distribution of sequences:
show_tsne_distribution(all_sequences, info);
show_pca_distribution(all_sequences, info);
show_cluster_distributions(info, all_sequences);

% training the SVM and reporting its accuracy:
[trainedClassifier_svm, validationAccuracy] = apply_svm_to_sequences([all_sequences, info.cluster]);
fprintf(newline);
disp(['Validation accuracy for linear SVM: ', num2str(validationAccuracy*100), '%']);

% using decision trees to get a sense of what is different between sequences of coughs and sneezes:
[trainedClassifier_tree, classificationTree] = ...
    interpret_clusters_using_decision_tree(info, all_full_sequences, ...
    all_full_sequences_non_normalized, n_peaks_to_take_from_side, ...
    use_normalized_sequences_for_training_classifier);

% saving all the important parameters so we can use them when predicting:
save('cough_sneeze_saved_info.mat', ...
    'smoothing_param', 'subsequence_to_take', 'zscore_sequences', ...
    'n_peaks_to_take_from_side', 'xlsx_sheet_names', ...
    'use_normalized_sequences_for_training_classifier', ...
    'trainedClassifier_tree', 'trainedClassifier_svm', ...
    'classificationTree', 'validationAccuracy', 'similarity_threshold');


function [trainedClassifier_tree, classificationTree] = ...
    interpret_clusters_using_decision_tree(info, all_full_sequences, ...
    all_full_sequences_non_normalized, n_peaks_to_take_from_side, ...
    use_normalized_sequences_for_training_classifier)

% calculating decision tree and using it to interprate the tree:
if use_normalized_sequences_for_training_classifier
    [all_points, all_full_sequences, info, all_global_max_times] = extract_critical_points(all_full_sequences, n_peaks_to_take_from_side, info);
    [interval_durations, global_max_val, slope] = extract_additional_features(all_full_sequences, all_global_max_times);
else
    [all_points, all_full_sequences_non_normalized, info, all_global_max_times] = extract_critical_points(all_full_sequences_non_normalized, n_peaks_to_take_from_side, info);
    [interval_durations, global_max_val, slope] = extract_additional_features(all_full_sequences_non_normalized, all_global_max_times);
end
[vals, var_names] = organize_training_data(all_points, interval_durations, global_max_val, slope, info);
[trainedClassifier_tree, validationAccuracy] = apply_tree_to_critical_points(vals, var_names, 4);
classificationTree = trainedClassifier_tree.ClassificationTree;
view(classificationTree, 'Mode', 'graph');
fprintf(newline);
disp(['Validation accuracy for Decision Tree = ', num2str(validationAccuracy*100), '%']);
fprintf(newline);
classificationTree.view;
fprintf(newline);
disp('Node number / error / size:');
disp([(1:length(classificationTree.NodeError))', classificationTree.NodeError, classificationTree.NodeSize]);

% plotting analysis of predictor importance:
important_predictors = classificationTree.PredictorNames(classificationTree.predictorImportance > 0)';
predictor_importance = nonzeros(classificationTree.predictorImportance);
[~, order] = sort(predictor_importance, 'descend');
important_predictors = important_predictors(order);
predictor_importance = predictor_importance(order);
fprintf(newline);
disp('Predictor importance:');
disp([num2cell(predictor_importance), important_predictors]);


function [vals, var_names] = organize_training_data(all_points, interval_durations, global_max_val, slope, info)

% reorganizing the points so we can use more features for the decision tree:
for i = 1 : length(all_points)
    all_points{i} = [ ...
        all_points{i}(:,1), ... % absolute time of peak
        [diff(all_points{i}(:,1)); 0], ... % time diff between curr peak and next peak
        [0; 0; all_points{i}(3:end,1) - all_points{i}(1:end-2,1)], ... % time diff between curr peak and two peaks earlier
        all_points{i}(:,2), ... % curr peak height
        [diff(all_points{i}(:,2)); 0], ... % next peak height minus curr peak height
        [0; 0; all_points{i}(1:end-2,2) ./ all_points{i}(3:end,2)]]; % height of two peaks ago divided by curr peak height
end

all_points = cellfun(@(x) reshape(x', [], 1), all_points, 'UniformOutput', false);

var_names = load_var_names;
var_names = cellfun(@transpose, var_names, 'UniformOutput', false);
var_names = [var_names{:}]';
var_names = var_names(:);
var_names{end+1} = 'class';

vals = cell2mat(all_points');
vals(end+1,:) = interval_durations ./ global_max_val;
vals(end+1,:) = interval_durations;
vals(end+1,:) = slope;
vals(end+1,:) = info.cluster == 2;
vals = vals';

var_names(end:end+3) = {'aspect_ratio', 'interval_durations', 'slope', 'class'};


function [interval_durations, global_max_val, slope] = extract_additional_features(all_full_sequences, all_global_max_times)

% calculating the duration of the above zero interval and the height of the global maximum:
interval_durations = nan(size(all_full_sequences,1),1);
global_max_val = nan(size(all_full_sequences,1),1);
slope = nan(size(all_full_sequences,1),1);
for i = 1 : size(all_full_sequences,1)
    cc = bwconncomp(all_full_sequences(i,:) > 0);
    cc_idx = cellfun(@(x) x(1) <= all_global_max_times(i) && x(end) >= all_global_max_times(i), cc.PixelIdxList);
    interval_durations(i) = length(cc.PixelIdxList{cc_idx});
    global_max_val(i) = all_full_sequences(i,all_global_max_times(i));
    slope_idx = cc.PixelIdxList{cc_idx}(end);
    slope(i) = diff(all_full_sequences(i,slope_idx:slope_idx+1));
end


function [all_sequences, all_full_sequences] = normalize_sequences(zscore_sequences, all_sequences, all_full_sequences)

% z-scoring the sequences:
if zscore_sequences
    seq_means = mean(all_sequences,2);
    seq_stds = std(all_sequences,1,2);
    all_full_sequences = (all_full_sequences - seq_means) ./ seq_stds;
    all_sequences = zscore(all_sequences')';
end


function [all_sequences, all_full_sequences] = extract_subsequences(all_sequences, subsequence_to_take)

% showing the raw sequences and the SD:
figure;
subplot(2,1,1);
plot(all_sequences');
axis square;
title('All sequences');
subplot(2,1,2);
plot(std(all_sequences));
axis square;
title('S.D.');

% taking the subsequence corresponding to the behaviors:
all_full_sequences = all_sequences;
all_sequences = all_sequences(:,subsequence_to_take);


function show_pca_distribution(all_sequences, info)

[~,all_sequences_pca] = pca(all_sequences, 'NumComponents', 3);
show_scatter3_distribution(all_sequences_pca, info, 'PCA');


function show_tsne_distribution(all_sequences, info)

% showing the sequence space using tsne:
all_sequences_tsne = tsne(all_sequences, 'NumDimensions', 3);
show_scatter3_distribution(all_sequences_tsne, info, 'tSNE');


function show_scatter3_distribution(all_sequences_dim_reduced, info, tit)

figure;

is_sneeze = info.cluster == 2;
is_cough = ~is_sneeze;

subplot(1,2,1);
scatter3(all_sequences_dim_reduced(is_cough,1), all_sequences_dim_reduced(is_cough,2), all_sequences_dim_reduced(is_cough,3), '.g');
hold on;
scatter3(all_sequences_dim_reduced(is_sneeze,1), all_sequences_dim_reduced(is_sneeze,2), all_sequences_dim_reduced(is_sneeze,3), '.r');
hold off;
axis image;
axis vis3d;
rotate3d on;
xlabel('X');
ylabel('Y');
zlabel('Z');
title([tit, ' 3 components']);
legend({'Cough', 'Sneeze'});

subplot(1,2,2);

is_pre = cellfun(@(x) ~isempty(x), regexp(info.Assay, 'Pre'));
is_post = cellfun(@(x) ~isempty(x), regexp(info.Assay, 'Post'));
is_cough_experiment = cellfun(@(x) ~isempty(x), regexp(info.Assay, 'Cough'));
is_sneeze_experiment = cellfun(@(x) ~isempty(x), regexp(info.Assay, 'Sneeze'));

scatter3( ...
    all_sequences_dim_reduced(is_pre & is_cough_experiment,1), ...
    all_sequences_dim_reduced(is_pre & is_cough_experiment,2), ...
    all_sequences_dim_reduced(is_pre & is_cough_experiment,3), 'or');

hold on;

scatter3( ...
    all_sequences_dim_reduced(is_post & is_cough_experiment,1), ...
    all_sequences_dim_reduced(is_post & is_cough_experiment,2), ...
    all_sequences_dim_reduced(is_post & is_cough_experiment,3), '^g');

scatter3( ...
    all_sequences_dim_reduced(is_pre & is_sneeze_experiment,1), ...
    all_sequences_dim_reduced(is_pre & is_sneeze_experiment,2), ...
    all_sequences_dim_reduced(is_pre & is_sneeze_experiment,3), '+b');

scatter3( ...
    all_sequences_dim_reduced(is_post & is_sneeze_experiment,1), ...
    all_sequences_dim_reduced(is_post & is_sneeze_experiment,2), ...
    all_sequences_dim_reduced(is_post & is_sneeze_experiment,3), '*k');

hold off;
axis image;
axis vis3d;
rotate3d on;
xlabel('X');
ylabel('Y');
zlabel('Z');
title([tit, ' 3 components']);
legend({'Pre - Cough Exp.', 'Post - Cough Exp.', 'Pre - Sneeze Exp.', 'Post - Sneeze Exp.'});


function show_cluster_distributions(info, all_sequences)

predicted_classes = info.cluster - 1;

% showing the distribution of clusters between the different experiments:
unq_animals = unique(info.Animal);
fprintf(newline);
for i = 1 : length(unq_animals)
    seq_to_take = ismember(info.Animal, unq_animals(i));
    disp(['Animal: ', num2str(unq_animals(i))]);
    disp(['    Cluster #1 Cough Pre: ', num2str(nnz(strcmp(info.Assay(seq_to_take), 'Cough_Pre') & predicted_classes(seq_to_take) == 0))]);
    disp(['    Cluster #1 Cough Post: ', num2str(nnz(strcmp(info.Assay(seq_to_take), 'Cough_Post') & predicted_classes(seq_to_take) == 0))]);
    disp(['    Cluster #2 Cough Pre: ', num2str(nnz(strcmp(info.Assay(seq_to_take), 'Cough_Pre') & predicted_classes(seq_to_take) == 1))]);
    disp(['    Cluster #2 Cough Post: ', num2str(nnz(strcmp(info.Assay(seq_to_take), 'Cough_Post') & predicted_classes(seq_to_take) == 1))]);
    disp(['    Cluster #1 Sneeze Pre: ', num2str(nnz(strcmp(info.Assay(seq_to_take), 'Sneeze_Pre') & predicted_classes(seq_to_take) == 0))]);
    disp(['    Cluster #1 Sneeze Post: ', num2str(nnz(strcmp(info.Assay(seq_to_take), 'Sneeze_Post') & predicted_classes(seq_to_take) == 0))]);
    disp(['    Cluster #2 Sneeze Pre: ', num2str(nnz(strcmp(info.Assay(seq_to_take), 'Sneeze_Pre') & predicted_classes(seq_to_take) == 1))]);
    disp(['    Cluster #2 Sneeze Post: ', num2str(nnz(strcmp(info.Assay(seq_to_take), 'Sneeze_Post') & predicted_classes(seq_to_take) == 1))]);
    fprintf(newline);
end
fprintf(newline);
predicted_classes = predicted_classes + 1;

% showing all clusters:
figure;
subplot(1,4,1);
plot(all_sequences(predicted_classes == 1,:)','Color', [0.7 0.7 0.7], 'LineWidth', 0.5);
hold on;
plot(mean(all_sequences(predicted_classes == 1,:),1'), 'Color', 'k', 'LineWidth', 2);
hold off;
axis square;
title(['Cluster #1 (n=', num2str(nnz(predicted_classes == 1)), ')']);
ylim([-4 4]);
xlim([1, size(all_sequences,2)]);
subplot(1,4,2);
plot(all_sequences(predicted_classes == 2,:)','Color', [0.7 0.7 0.7], 'LineWidth', 0.5);
hold on;
plot(mean(all_sequences(predicted_classes == 2,:),1'), 'Color', 'k', 'LineWidth', 2);
hold off;
axis square;
title(['Cluster #2 (n=', num2str(nnz(predicted_classes == 2)), ')']);
ylim([-4 4]);
xlim([1, size(all_sequences,2)]);
subplot(1,4,3);
cluster_1_sequences = all_sequences(predicted_classes == 1,:);
plot(mean(cluster_1_sequences)', 'r', 'LineWidth', 1);
hold on;
sampled_times = 10:10:size(cluster_1_sequences,2);
cluster_1_errors = std(cluster_1_sequences(:,sampled_times));
errorbar(sampled_times, mean(cluster_1_sequences(:,sampled_times)), cluster_1_errors, 'r', 'LineStyle', 'none');
cluster_2_sequences = all_sequences(predicted_classes == 2,:);
plot(mean(cluster_2_sequences)', 'k', 'LineWidth', 1);
sampled_times = 8:10:size(cluster_1_sequences,2);
cluster_2_errors = std(cluster_2_sequences(:,sampled_times));
errorbar(sampled_times, mean(cluster_2_sequences(:,sampled_times)), cluster_2_errors, 'k', 'LineStyle', 'none');
hold off;
legend({'Cluster #1 - Avg.', 'Cluster #1 - S.D.', 'Cluster #2 - Avg.', 'Cluster #2 - S.D.'}, 'Location', 'southwest');
axis square;
title('Both sequences');
ylim([-4 4]);
xlim([1, size(all_sequences,2)]);

% showing clusters as scatter plots:
[~, scores] = pca(all_sequences, 'NumComponents', 3);
subplot(1,4,4);
plot(scores(predicted_classes == 1,1), scores(predicted_classes == 1,2), '*r');
hold on;
plot(scores(predicted_classes == 2,1), scores(predicted_classes == 2,2), '*k');
hold off;
xlabel('PC1');
ylabel('PC2');
title('Sequences PCA');
legend({'Cluster #1', 'Cluster #2'}, 'Location', 'southwest');
axis image;
xlim([min(scores(:,1))-1, max(scores(:,1))+1]);
ylim([min(scores(:,2))-1, max(scores(:,2))+1]);


function var_names = load_var_names

% names of variables to help interpret the decision tree.
% this function should never be modified!

var_names{1,1} = { ...
    'max -4 time', ...
    'min -4 time', ...
    'max -3 time', ...
    'min -3 time', ...
    'max -2 time', ...
    'min -2 time', ...
    'max -1 time', ...
    'min -1 time', ...
    'max 0 time', ...
    'min 0 time', ...
    'max 1 time', ...
    'min 1 time', ...
    'max 2 time', ...
    'min 2 time', ...
    'max 3 time', ...
    'min 3 time', ...
    'max 4 time'};

var_names{1,2} = { ...
    'max -4 min -4 time diff', ...
    'min -4 max -3 time diff', ...
    'max -3 min -3 time diff', ...
    'min -3 max -2 time diff', ...
    'max -2 min -2 time diff', ...
    'min -2 max -1 time diff', ...
    'max -1 min -1 time diff', ...
    'min -1 max 0 time diff', ...
    'max 0 min 0 time diff', ...
    'min 0 max 1 time diff', ...
    'max 1 min 1 time diff', ...
    'min 1 max 2 time diff', ...
    'max 2 min 2 time diff', ...
    'min 2 max 3 time diff', ...
    'max 3 min 3 time diff', ...
    'min 3 max 4 time diff', ...
    '(none time diff)'};

var_names{1,3} = { ...
    '(none time diff 1)', ...
    '(none time diff 2)', ...
    'max -4 max -3 time diff', ...
    'min -4 min -3 time diff', ...
    'max -3 max -2 time diff', ...
    'min -3 min -2 time diff', ...
    'max -2 max -1 time diff', ...
    'min -2 min -1 time diff', ...
    'max -1 max 0 time diff', ...
    'min -1 min 0 time diff', ...
    'max 0 max 1 time diff', ...
    'min 0 min 1 time diff', ...
    'max 1 max 2 time diff', ...
    'min 1 min 2 time diff', ...
    'max 2 max 3 time diff', ...
    'min 2 min 3 time diff', ...
    'max 3 max 4 time diff'};

var_names{1,4} = { ...
    'max -4 height', ...
    'min -4 height', ...
    'max -3 height', ...
    'min -3 height', ...
    'max -2 height', ...
    'min -2 height', ...
    'max -1 height', ...
    'min -1 height', ...
    'max 0 height', ...
    'min 0 height', ...
    'max 1 height', ...
    'min 1 height', ...
    'max 2 height', ...
    'min 2 height', ...
    'max 3 height', ...
    'min 3 height', ...
    'max 4 height'};

var_names{1,5} = { ...
    'max -4 min -4 height diff', ...
    'min -4 max -3 height diff', ...
    'max -3 min -3 height diff', ...
    'min -3 max -2 height diff', ...
    'max -2 min -2 height diff', ...
    'min -2 max -1 height diff', ...
    'max -1 min -1 height diff', ...
    'min -1 max 0 height diff', ...
    'max 0 min 0 height diff', ...
    'min 0 max 1 height diff', ...
    'max 1 min 1 height diff', ...
    'min 1 max 2 height diff', ...
    'max 2 min 2 height diff', ...
    'min 2 max 3 height diff', ...
    'max 3 min 3 height diff', ...
    'min 3 max 4 height diff', ...
    '(none height diff)'};

var_names{1,6} = { ...
    '(none height diff 1)', ...
    '(none height diff 2)', ...
    'max -4 max -3 height ratio', ...
    'min -4 min -3 height ratio', ...
    'max -3 max -2 height ratio', ...
    'min -3 min -2 height ratio', ...
    'max -2 max -1 height ratio', ...
    'min -2 min -1 height ratio', ...
    'max -1 max 0 height ratio', ...
    'min -1 min 0 height ratio', ...
    'max 0 max 1 height ratio', ...
    'min 0 min 1 height ratio', ...
    'max 1 max 2 height ratio', ...
    'min 1 min 2 height ratio', ...
    'max 2 max 3 height ratio', ...
    'min 2 min 3 height ratio', ...
    'max 3 max 4 height ratio'};


function [all_points, all_sequences, info, all_global_max_times] = ...
    extract_critical_points(all_sequences, n_peaks_to_take_from_side, info)

% finding local min and max:
all_points = cell(size(all_sequences,1),1);
all_global_max_times = nan(size(all_sequences,1),1);
for i = 1 : size(all_sequences,1)
    local_max = find(all_sequences(i,2:end-1) > all_sequences(i,1:end-2) & all_sequences(i,2:end-1) > all_sequences(i,3:end))' + 1;
    pre_transition_peaks = local_max(local_max <= 500);
    global_max_time = pre_transition_peaks(end);
    all_global_max_times(i) = global_max_time;

    local_max(:,2) = all_sequences(i,local_max);
    local_min = find(all_sequences(i,2:end-1) < all_sequences(i,1:end-2) & all_sequences(i,2:end-1) < all_sequences(i,3:end))' + 1;
    local_min(:,2) = all_sequences(i,local_min);

    all_points{i} = [local_max; local_min];
    all_points{i} = sortrows(all_points{i});

    global_max_idx = find(global_max_time == all_points{i}(:,1));

    to_take = global_max_idx - n_peaks_to_take_from_side : global_max_idx + n_peaks_to_take_from_side;
    if any(to_take < 1 | to_take > size(all_points{i},1))
        all_points{i} = nan(n_peaks_to_take_from_side*2+1,2);
    else
        all_points{i} = all_points{i}(to_take,:);
        all_points{i}(:,1) = all_points{i}(:,1) - global_max_time;
    end
end

% removing sequences that didn't have enough peaks around the center:
to_remove = cellfun(@(x) isnan(x(1)), all_points);
if any(to_remove)
    all_sequences(to_remove,:) = [];
    all_points(to_remove) = [];
    info = structfun(@(x) x(~to_remove), info, 'UniformOutput', false);
end
clear to_remove;


function [all_sequences, info] = load_and_prepare_data(filename, smoothing_param, xlsx_sheet_names)

% loading sequences info:
[~, ~, raw_info] = xlsread(filename, xlsx_sheet_names{1}); %#ok<XLSRD>
info.Animal = cell2mat(raw_info(3:end, strcmp(raw_info(2,:), 'Animal')));
info.Assay = raw_info(3:end, strcmp(raw_info(2,:), 'Assay'));

% loading sequences:
all_sequences = xlsread(filename, xlsx_sheet_names{2})'; %#ok<XLSRD>

% identifying repeating sequences and removing them:
continue_removing = 1;
n_removed_sequences = 0;
while continue_removing
    seq_distances = squareform(pdist(all_sequences));
    [x,y] = find(seq_distances == 0);
    to_remove = x == y;
    x(to_remove) = [];
    y(to_remove) = [];
    repeating_sequences = unique(sort([x,y], 2), 'rows');
    to_keep = 1 : size(all_sequences,1);
    to_keep(unique(repeating_sequences(:,2))) = [];
    info = structfun(@(x) x(to_keep), info, 'UniformOutput', false);
    all_sequences = all_sequences(to_keep,:);
    continue_removing = ~isempty(repeating_sequences);
    n_removed_sequences = n_removed_sequences + length(unique(repeating_sequences(:,2)));
end
disp(['Total removed repeating sequences = ', num2str(n_removed_sequences)]);

% smoothing sequences:
if smoothing_param > 0
    ft = fittype( 'smoothingspline' );
    opts = fitoptions( 'Method', 'SmoothingSpline' );
    opts.SmoothingParam = smoothing_param;
    x = (1:size(all_sequences,2))';
    for i = 1 : size(all_sequences,1)
        fitresult = fit(x, all_sequences(i,:)', ft, opts);
        all_sequences(i,:) = fitresult(x);
    end
    clear x;
end


function [trainedClassifier, validationAccuracy] = apply_svm_to_sequences(trainingData)

% Convert input to table
inputTable = array2table(trainingData, 'VariableNames', {'column_1', 'column_2', 'column_3', 'column_4', 'column_5', 'column_6', 'column_7', 'column_8', 'column_9', 'column_10', 'column_11', 'column_12', 'column_13', 'column_14', 'column_15', 'column_16', 'column_17', 'column_18', 'column_19', 'column_20', 'column_21', 'column_22', 'column_23', 'column_24', 'column_25', 'column_26', 'column_27', 'column_28', 'column_29', 'column_30', 'column_31', 'column_32', 'column_33', 'column_34', 'column_35', 'column_36', 'column_37', 'column_38', 'column_39', 'column_40', 'column_41', 'column_42', 'column_43', 'column_44', 'column_45', 'column_46', 'column_47', 'column_48', 'column_49', 'column_50', 'column_51', 'column_52', 'column_53', 'column_54', 'column_55', 'column_56', 'column_57', 'column_58', 'column_59', 'column_60', 'column_61', 'column_62', 'column_63', 'column_64', 'column_65', 'column_66', 'column_67', 'column_68', 'column_69', 'column_70', 'column_71', 'column_72', 'column_73', 'column_74', 'column_75', 'column_76', 'column_77', 'column_78', 'column_79', 'column_80', 'column_81', 'column_82', 'column_83', 'column_84', 'column_85', 'column_86', 'column_87', 'column_88', 'column_89', 'column_90', 'column_91', 'column_92', 'column_93', 'column_94', 'column_95', 'column_96', 'column_97', 'column_98', 'column_99', 'column_100', 'column_101', 'column_102', 'column_103', 'column_104', 'column_105', 'column_106', 'column_107', 'column_108', 'column_109', 'column_110', 'column_111', 'column_112', 'column_113', 'column_114', 'column_115', 'column_116', 'column_117', 'column_118', 'column_119', 'column_120', 'column_121', 'column_122', 'column_123', 'column_124', 'column_125', 'column_126', 'column_127', 'column_128', 'column_129', 'column_130', 'column_131', 'column_132', 'column_133', 'column_134', 'column_135', 'column_136', 'column_137', 'column_138', 'column_139', 'column_140', 'column_141', 'column_142', 'column_143', 'column_144', 'column_145', 'column_146', 'column_147', 'column_148', 'column_149', 'column_150', 'column_151', 'column_152', 'column_153', 'column_154', 'column_155', 'column_156', 'column_157'});

predictorNames = {'column_1', 'column_2', 'column_3', 'column_4', 'column_5', 'column_6', 'column_7', 'column_8', 'column_9', 'column_10', 'column_11', 'column_12', 'column_13', 'column_14', 'column_15', 'column_16', 'column_17', 'column_18', 'column_19', 'column_20', 'column_21', 'column_22', 'column_23', 'column_24', 'column_25', 'column_26', 'column_27', 'column_28', 'column_29', 'column_30', 'column_31', 'column_32', 'column_33', 'column_34', 'column_35', 'column_36', 'column_37', 'column_38', 'column_39', 'column_40', 'column_41', 'column_42', 'column_43', 'column_44', 'column_45', 'column_46', 'column_47', 'column_48', 'column_49', 'column_50', 'column_51', 'column_52', 'column_53', 'column_54', 'column_55', 'column_56', 'column_57', 'column_58', 'column_59', 'column_60', 'column_61', 'column_62', 'column_63', 'column_64', 'column_65', 'column_66', 'column_67', 'column_68', 'column_69', 'column_70', 'column_71', 'column_72', 'column_73', 'column_74', 'column_75', 'column_76', 'column_77', 'column_78', 'column_79', 'column_80', 'column_81', 'column_82', 'column_83', 'column_84', 'column_85', 'column_86', 'column_87', 'column_88', 'column_89', 'column_90', 'column_91', 'column_92', 'column_93', 'column_94', 'column_95', 'column_96', 'column_97', 'column_98', 'column_99', 'column_100', 'column_101', 'column_102', 'column_103', 'column_104', 'column_105', 'column_106', 'column_107', 'column_108', 'column_109', 'column_110', 'column_111', 'column_112', 'column_113', 'column_114', 'column_115', 'column_116', 'column_117', 'column_118', 'column_119', 'column_120', 'column_121', 'column_122', 'column_123', 'column_124', 'column_125', 'column_126', 'column_127', 'column_128', 'column_129', 'column_130', 'column_131', 'column_132', 'column_133', 'column_134', 'column_135', 'column_136', 'column_137', 'column_138', 'column_139', 'column_140', 'column_141', 'column_142', 'column_143', 'column_144', 'column_145', 'column_146', 'column_147', 'column_148', 'column_149', 'column_150', 'column_151', 'column_152', 'column_153', 'column_154', 'column_155', 'column_156'};
predictors = inputTable(:, predictorNames);
response = inputTable.column_157;
classNames = [1; 2];

% Train a classifier
% This code specifies all the classifier options and trains the classifier.
classificationSVM = fitcsvm(...
    predictors, ...
    response, ...
    'KernelFunction', 'linear', ...
    'PolynomialOrder', [], ...
    'KernelScale', 'auto', ...
    'BoxConstraint', 1, ...
    'Standardize', true, ...
    'ClassNames', classNames);

% Create the result struct with predict function
predictorExtractionFcn = @(x) array2table(x, 'VariableNames', predictorNames);
svmPredictFcn = @(x) predict(classificationSVM, x);
trainedClassifier.predictFcn = @(x) svmPredictFcn(predictorExtractionFcn(x));

% Add additional fields to the result struct
trainedClassifier.ClassificationSVM = classificationSVM;
trainedClassifier.About = 'This struct is a trained model exported from Classification Learner R2023b.';
trainedClassifier.HowToPredict = sprintf('To make predictions on a new predictor column matrix, X, use: \n  [yfit,scores] = c.predictFcn(X) \nreplacing ''c'' with the name of the variable that is this struct, e.g. ''trainedModel''. \n \nX must contain exactly 156 columns because this model was trained using 156 predictors. \nX must contain only predictor columns in exactly the same order and format as your training \ndata. Do not include the response column or any columns you did not import into the app. \n \nFor more information, see <a href="matlab:helpview(fullfile(docroot, ''stats'', ''stats.map''), ''appclassification_exportmodeltoworkspace'')">How to predict using an exported model</a>.');

% Perform cross-validation
partitionedModel = crossval(trainedClassifier.ClassificationSVM, 'KFold', 5);

% Compute validation accuracy
validationAccuracy = 1 - kfoldLoss(partitionedModel, 'LossFun', 'ClassifError');


function L = get_branch_leaves(Z)

% this function finds the list of all sequences that are under each branch in the hierarchical clustering tree.
% it basically just calls the recursive function that does the work.

L = cell(size(Z,1),2);

for idx = 1 : size(Z,1)
    L(idx,:) = get_branch_leaves_recursive_main(Z,idx);
end


function L = get_branch_leaves_recursive_main(Z, idx)

% this recursive function (called by 'get_branch_leaves') finds the list of
% all sequences under each branch in the hierarchical clustering tree:

L = cell(1,2);

if Z(idx,1) <= size(Z,1)+1
    L{1} = Z(idx,1);
else
    L{1} = get_branch_leaves_recursive_main(Z, Z(idx,1)-size(Z,1)-1);
    L{1} = cell2mat(L{1});
end

if Z(idx,2) <= size(Z,1)+1
    L{2} = Z(idx,2);
else
    L{2} = get_branch_leaves_recursive_main(Z, Z(idx,2)-size(Z,1)-1);
    L{2} = cell2mat(L{2});
end


function [trainedClassifier, validationAccuracy] = apply_tree_to_critical_points(trainingData, var_names, max_splits)

inputTable = array2table(trainingData, 'VariableNames', var_names');

predictorNames = var_names(1:end-1)';
predictors = inputTable(:, predictorNames);
response = inputTable.class;
classNames = [0; 1];

% Train a classifier
% This code specifies all the classifier options and trains the classifier.
classificationTree = fitctree(...
    predictors, ...
    response, ...
    'SplitCriterion', 'gdi', ...
    'MaxNumSplits', max_splits, ...
    'Surrogate', 'off', ...
    'ClassNames', classNames);

% Create the result struct with predict function
predictorExtractionFcn = @(x) array2table(x, 'VariableNames', predictorNames);
treePredictFcn = @(x) predict(classificationTree, x);
trainedClassifier.predictFcn = @(x) treePredictFcn(predictorExtractionFcn(x));

% Add additional fields to the result struct
trainedClassifier.ClassificationTree = classificationTree;
trainedClassifier.About = 'This struct is a trained model exported from Classification Learner R2023b.';
trainedClassifier.HowToPredict = sprintf('To make predictions on a new predictor column matrix, X, use: \n  [yfit,scores] = c.predictFcn(X) \nreplacing ''c'' with the name of the variable that is this struct, e.g. ''trainedModel''. \n \nX must contain exactly 105 columns because this model was trained using 105 predictors. \nX must contain only predictor columns in exactly the same order and format as your training \ndata. Do not include the response column or any columns you did not import into the app. \n \nFor more information, see <a href="matlab:helpview(fullfile(docroot, ''stats'', ''stats.map''), ''appclassification_exportmodeltoworkspace'')">How to predict using an exported model</a>.');

% Perform cross-validation
partitionedModel = crossval(trainedClassifier.ClassificationTree, 'KFold', 5);

% Compute validation accuracy
validationAccuracy = 1 - kfoldLoss(partitionedModel, 'LossFun', 'ClassifError');


function info = show_hierarchical_clusters(all_sequences, info, similarity_threshold, cough_cluster_branches)

% calculating clusters:
Y = pdist(all_sequences, 'euclidean');     % Compute pairwise distances using Euclidean distance
Z = linkage(Y, 'weighted');     % Create hierarchical cluster tree using average linkage
L = get_branch_leaves(Z);
L = cellfun(@(x1,x2) [x1,x2], L(:,1), L(:,2), 'UniformOutput', false);
is_pre = ~cellfun(@isempty, regexp(info.Assay, 'Pre'));
mean_is_pre = cellfun(@(x) mean(is_pre(x)), L);
num_is_pre = cellfun(@(x) nnz(is_pre(x)), L);
n_leaves = cellfun(@(x) length(x), L);

% plotting the clustering:
figure('Name', 'Hierarchical clustering:');
[H, ~, outperm] = dendrogram(Z, size(all_sequences,1), 'ColorThreshold', similarity_threshold); % Plot dendrogram for all observations and get the output permutation
is_cough_experiment = ~cellfun(@isempty, regexp(info.Assay, 'Cough'));
x_label = num2str(sum(double([is_pre, info.Animal - 9910]) .* [10 1], 2), '%.2d');
x_label = mat2cell(x_label, ones(size(x_label,1),1), 2);
set(gca, 'XTick', 1:length(outperm), 'XTickLabel', x_label(outperm), 'XTickLabelRotation', 90, 'FontSize', 6, 'FontName', 'courier');
lines_y = cell2mat({H.YData}');
y_positions_to_take = [ ...
    lines_y(prod(lines_y(:,1:2) - similarity_threshold, 2) < 0, 1); ...
    lines_y(prod(lines_y(:,3:4) - similarity_threshold, 2) < 0, 4)];
to_plot = find(ismember(lines_y(:,2), y_positions_to_take));
additional_branches = find(lines_y(:,2) >= similarity_threshold);

disp([newline, 'For each of the branches above the "similarity_threshold" set in cough_sneeze_train.m']);
disp(['we plot the following 2x2 table:', newline]);
disp('                  |  Pre_Treatment   | Post_Treatment');
disp('------------------------------------------------------');
disp('Cough_Experiment  |                  |');
disp('------------------------------------------------------');
disp(['Sneeze_Experiment |                  |', newline]);

for j = 1 : length(H)

    if isequal(H(j).Color, [0,0,0])
        H(j).Color = [0.7 0.7 0.7];
    end

    if ismember(j, to_plot) || ismember(j, additional_branches)

        % labeling the tree:
        text(mean(H(j).XData(2:3)), H(j).YData(2)+0.05, ...
            [num2str(round(mean_is_pre(j)*100)), '%', newline, '(', num2str(num_is_pre(j)), '/', num2str(n_leaves(j)), ...
            '; C', num2str(j), ')'], ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 6);

        % showing the distribution of sequences in the different clusters:
        curr_leaves = L{j};
        pre_and_cough_exp = nnz(is_pre(curr_leaves) & is_cough_experiment(curr_leaves));
        post_and_cough_exp = nnz(~is_pre(curr_leaves) & is_cough_experiment(curr_leaves));
        pre_and_sneeze_exp = nnz(is_pre(curr_leaves) & ~is_cough_experiment(curr_leaves));
        post_and_sneeze_exp = nnz(~is_pre(curr_leaves) & ~is_cough_experiment(curr_leaves));
        output_mat = [pre_and_cough_exp, pre_and_sneeze_exp; post_and_cough_exp, post_and_sneeze_exp];
        disp(['C', num2str(j), ':']);
        disp(output_mat);
        fprintf(newline);
    end
end

% plotting the sequence clusters:
n_blocks = ceil(sqrt(length(to_plot)));
figure;
for j = 1 : length(to_plot)
    subplot(n_blocks, n_blocks, j);
    plot(all_sequences(cell2mat(L(to_plot(j),:)),:)', 'Color', [0.7 0.7 0.7], 'LineWidth', 0.5);
    hold on;
    plot(mean(all_sequences(cell2mat(L(to_plot(j),:)),:),1)', 'Color', 'k', 'LineWidth', 2);
    hold off;
    title(['C', num2str(to_plot(j)), newline, ...
        num2str(round(mean_is_pre(to_plot(j))*100)), '%', ' (', num2str(num_is_pre(to_plot(j))), '/', num2str(n_leaves(to_plot(j))), ')']);
end

% creating the labels of the clusters:
sneeze_cluster_branches = setdiff(to_plot, cough_cluster_branches);
info.cluster(:) = nan;
info.cluster(cell2mat(L(cough_cluster_branches)')') = 1;
info.cluster(cell2mat(L(sneeze_cluster_branches)')') = 2;
info.cluster = info.cluster(:);
