function cough_sneeze_predict

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% This code was developed as a part of the "A vagal-brainstem interoceptive
% circuit for cough-like defensive behaviors in mice" (Gannot et al.) by
% Tomer Stern, University of Michigan, Ann Arbor, email: tomers@umich.edu
% This code is called after 'cough_sneeze_train.m', and its purpose is to load
% a new set of respiratory behaviors of a mouse and classify them into 'cough'
% or 'sneeze', based on a linear SVM that was trained in 'cough_sneeze_train'.
% It needs no input arguments, as it loads all the parameters that were
% already set during the call to 'cough_sneeze_train.m'.
% the output is then plotted both as color-coded scatter plots (tSNE and PCA),
% on the hierarchical clustering tree of the sequences, and into a new Excel file
% with an added column showing the inferred clusters of the sequences.

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

% loading script parameters (these were saved while running 'cough_sheeze_train')
saved_classifier_model_filename = 'cough_sneeze_saved_info.mat';
if isfile(saved_classifier_model_filename)
    load(saved_classifier_model_filename); %#ok<LOAD>
else
    disp('The file "cough_sneeze_saved_info.mat" cannot be found.');
    disp('Try running the script "cough_sneeze_train.m" first.');
    return;
end

% browsing for the name of the Excel file with the unseen sequences:
[file,path] = uigetfile('*.xlsx', 'Please select the *.xlsx new sequences file:');
filename = fullfile(path, file);
clear file path;

% loading sequences data:
[all_sequences, info, raw_info, raw_sequences] = load_and_prepare_data(filename, smoothing_param, xlsx_sheet_names); %#ok<*USENS>

% extracting the subsequences of interest (the range is set at 'cough_sneeze_train'):
[all_sequences, all_full_sequences] = extract_subsequences(all_sequences, subsequence_to_take);

% Z-score normalizing the sequences and subsequences:
all_full_sequences_non_normalized = all_full_sequences; %#ok<NASGU>
[all_sequences, all_full_sequences] = normalize_sequences(zscore_sequences, all_sequences, all_full_sequences); %#ok<ASGLU>

% applying the trained classifier on the subsequences:
info.cluster = trainedClassifier_svm.predictFcn(all_sequences);

% showing the similarity space of the sequences with their class identity:
show_tsne_distribution(all_sequences, info);
show_pca_distribution(all_sequences, info);
show_cluster_distributions(info, all_sequences);

% just since we have the code for it - showing hierarchical clustering of these unseen sequences:
show_hierarchical_clustering(all_sequences, info, similarity_threshold);

% preparing output data to be saved into a new Excel file and saving it:
output_filename = regexprep(filename, '\.xlsx$', ' (classes added).xlsx');
if isfile(output_filename)
    delete(output_filename);
end
added_class = [{[]}; {'Predicted Class'}; num2cell(info.cluster)];
raw_info = [raw_info, added_class];
warning('off', 'MATLAB:xlswrite:AddSheet');
xlswrite(output_filename, raw_info, xlsx_sheet_names{1}); %#ok<*XLSWT>
xlswrite(output_filename, raw_sequences, xlsx_sheet_names{2});
warning('on', 'MATLAB:xlswrite:AddSheet');
disp(['Output file name: "', output_filename, '"', newline]);
disp('Done.');


function [all_sequences, all_full_sequences] = normalize_sequences(zscore_sequences, all_sequences, all_full_sequences)

% z-scoring the sequences:
if zscore_sequences
    seq_means = mean(all_sequences,2);
    seq_stds = std(all_sequences,1,2);
    all_full_sequences = (all_full_sequences - seq_means) ./ seq_stds;
    all_sequences = zscore(all_sequences')';
end


function [all_sequences, all_full_sequences] = extract_subsequences(all_sequences, subsequence_to_take)

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

scatter3(all_sequences_dim_reduced(is_cough,1), all_sequences_dim_reduced(is_cough,2), all_sequences_dim_reduced(is_cough,3), 'og');
hold on;
scatter3(all_sequences_dim_reduced(is_sneeze,1), all_sequences_dim_reduced(is_sneeze,2), all_sequences_dim_reduced(is_sneeze,3), 'or');
hold off;
axis image;
axis vis3d;
rotate3d on;
xlabel('X');
ylabel('Y');
zlabel('Z');
title([tit, ' 3 components']);
legend({'Cough', 'Sneeze'});


function show_cluster_distributions(info, all_sequences)

% showing all clusters:
figure;
subplot(1,4,1);
plot(all_sequences(info.cluster == 1,:)');
axis square;
title(['Cluster #1 (n=', num2str(nnz(info.cluster == 1)), ')']);
ylim([-4 4]);
xlim([1, size(all_sequences,2)]);
subplot(1,4,2);
plot(all_sequences(info.cluster == 2,:)');
axis square;
title(['Cluster #2 (n=', num2str(nnz(info.cluster == 2)), ')']);
ylim([-4 4]);
xlim([1, size(all_sequences,2)]);
subplot(1,4,3);
cluster_1_sequences = all_sequences(info.cluster == 1,:);
plot(mean(cluster_1_sequences)', 'r', 'LineWidth', 1);
hold on;
sampled_times = 10:10:size(cluster_1_sequences,2);
cluster_1_errors = std(cluster_1_sequences(:,sampled_times));
errorbar(sampled_times, mean(cluster_1_sequences(:,sampled_times)), cluster_1_errors, 'r', 'LineStyle', 'none');
cluster_2_sequences = all_sequences(info.cluster == 2,:);
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
plot(scores(info.cluster == 1,1), scores(info.cluster == 1,2), '*r');
hold on;
plot(scores(info.cluster == 2,1), scores(info.cluster == 2,2), '*k');
hold off;
xlabel('PC1');
ylabel('PC2');
title('Sequences PCA');
legend({'Cluster #1', 'Cluster #2'}, 'Location', 'southwest');
axis image;
xlim([min(scores(:,1))-1, max(scores(:,1))+1]);
ylim([min(scores(:,2))-1, max(scores(:,2))+1]);


function [all_sequences, info, raw_info, raw_sequences] = load_and_prepare_data(filename, smoothing_param, xlsx_sheet_names)

% loading sequences info:
[~, ~, raw_info] = xlsread(filename, xlsx_sheet_names{1}); %#ok<XLSRD>
info.Number = cell2mat(raw_info(3:end, strcmp(raw_info(2,:), 'Number')));

% loading sequences:
[all_sequences, ~, raw_sequences] = xlsread(filename, xlsx_sheet_names{2}); %#ok<XLSRD>
all_sequences = all_sequences';

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
    raw_sequences = raw_sequences(:,to_keep);
    raw_info = raw_info([1;2;to_keep'+2],:);
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


function show_hierarchical_clustering(all_sequences, info, similarity_threshold)

Y = pdist(all_sequences, 'euclidean');     % Compute pairwise distances using Euclidean distance
Z = linkage(Y, 'weighted');     % Create hierarchical cluster tree using average linkage

% plotting the clustering:
figure('Name', 'Hierarchical clustering:');
[H, ~, outperm] = dendrogram(Z, size(all_sequences,1), 'ColorThreshold', similarity_threshold); % Plot dendrogram for all observations and get the output permutation
x_label = [num2str(info.Number, '%.3d'), repmat(',', length(info.cluster), 1), num2str(info.cluster)];
x_label = mat2cell(x_label, ones(size(x_label,1),1), size(x_label,2));
set(gca, 'XTick', 1:length(outperm), 'XTickLabel', x_label(outperm), 'XTickLabelRotation', 90, 'FontSize', 6, 'FontName', 'courier');

for j = 1 : length(H)
    if isequal(H(j).Color, [0,0,0])
        H(j).Color = [0.7 0.7 0.7];
    end
end

