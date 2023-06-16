# %%
import pandas as pd
import numpy as np

dataset = "youtube" # specify the task name 
save_name =  "sample_youtube" # results name
metric = "loss" # metric name
num_task = 100
result_df = [] # name of results files
results_dir = [
    f"../results/{save_name}_v1/{save_name}_v1_{dataset}.csv",
    f"../results/{save_name}_v2/{save_name}_v2_{dataset}.csv",
    f"../results/{save_name}_v3/{save_name}_v3_{dataset}.csv",
    f"../results/{save_name}_v4/{save_name}_v4_{dataset}.csv",
]
for result_dir in results_dir:
    tmp_result_df = pd.read_csv(result_dir, index_col=0)
    if len(result_df) == 0:
        result_df = tmp_result_df
    else:
        result_df = pd.concat([result_df, tmp_result_df], ignore_index = True)

# %%
subset_num = 2000

trained_subsets = set()
for trained_subset in result_df["Trained on"].values:
    trained_subsets.add(trained_subset)
trained_subsets = list(trained_subsets)

training_subsets = trained_subsets[:subset_num]

train_mask = [val in training_subsets for val in result_df["Trained on"].values]
train_result_df = result_df[train_mask]
# %%
from sklearn import linear_model

def fit_by_averaging(features, targets):
    targets = np.expand_dims(targets, axis=1)
    task_performances = features * targets
    task_performances = np.sum(task_performances, axis=0)
    counts = np.sum(features, axis=0)
    avg_task_performance = task_performances / counts
    # print(" ".join([str(task+1) for task in np.argsort(avg_task_performance)]))
    return avg_task_performance

""" Clustering """
task_models = []; onehot_features = []; avg_task_models = []
for task_id in range(num_task):
    task_df = train_result_df[train_result_df["Task"] == task_id]

    target_metric = f"valid_{metric}"
    sampled_tasks = task_df["Trained on"].values

    features = []
    targets = []
    for i, subsample in enumerate(sampled_tasks):
        # convert subsample from str to list
        sample_task = subsample.strip('][').split(' ')
        sample_task = [int(task) for task in sample_task if task ]

        sample_feature = np.zeros(shape=(1, num_task))
        sample_feature[0, sample_task] = 1
        tmp_target = task_df[task_df["Trained on"] == subsample][target_metric].values[0]
        if np.isnan(tmp_target):
            continue
        features.append(sample_feature)
        targets.append(tmp_target)
    features = np.concatenate(features, axis=0)
    targets = np.array(targets)

    if "loss" in target_metric or "Loss" in target_metric:
        targets = -targets
    else:
        targets = targets

    print(features.shape, targets.shape)

    # normalize the targets 
    mean, std = np.mean(targets), np.std(targets)
    targets = (targets - mean)/(std+1e-3)
    clf = linear_model.Ridge(alpha=1e-2, fit_intercept = True)
    clf.fit(features, targets)
    
    # averaging
    avg_task_performance = fit_by_averaging(features, targets)
    avg_task_performance[task_id] = 0
    task_models.append(clf.coef_)
    avg_task_models.append(avg_task_performance)

task_models = np.array(avg_task_models)
task_models[np.arange(num_task), np.arange(num_task)] = 0
sym_task_models = (task_models + task_models.T)/2


# %%
'''Spectral clustering '''
from sklearn.cluster import SpectralClustering

n_clusters = 20

clustering = SpectralClustering(
    n_clusters=n_clusters,
    affinity="precomputed", 
    n_init=100).fit(sym_task_models)

shuffle_idxes = np.concatenate([
    np.arange(num_task)[clustering.labels_ == i] for i in range(n_clusters)
    ])

groups = []
for i in range(n_clusters):
    group = np.arange(num_task)[clustering.labels_ == i]
    groups.append(group)
    print(group)

# %%
''' 
Extended Spectral Coclustering:
[(\Theta + \Theta.T)/2, \Theta,
 \Theta.T, 0      ]
'''
A_1 = np.concatenate([(task_models+task_models.T)/2, task_models], axis=1)
A_2 = np.concatenate([task_models.T, np.zeros_like(task_models)], axis=1)
A = np.concatenate([A_1, A_2], axis=0)

n_clusters = 21
clustering = SpectralClustering(
    n_clusters=n_clusters,
    affinity="precomputed", 
    n_init=100).fit(A)

groups = []
for i in range(n_clusters):
    group = np.arange(num_task*2)[clustering.labels_ == i]
    new_group = set(); has_target = False
    for task_id in group:
        if task_id<num_task:
            new_group.add(task_id)
            has_target = True
        else:
            new_group.add(task_id - num_task)
    if has_target:
        groups.append(np.array(list(new_group)))
print("Group length: {}".format(len(groups)))
for group in groups:
    print(group)

# %%
'''Spectral Coclustering'''

from sklearn.cluster import SpectralCoclustering

n_clusters = 20
clf = SpectralCoclustering(n_clusters=n_clusters, random_state=0, n_init=100).fit(task_models)

row_clusters = clf.row_labels_
col_clusters = clf.column_labels_

row_shuffle_idxes = np.concatenate([
    np.arange(num_task)[row_clusters == i] for i in range(n_clusters)
    ])

col_shuffle_idxes = np.concatenate([
    np.arange(num_task)[col_clusters == i] for i in range(n_clusters)
    ])

print("Groups: ")
for i in range(n_clusters):
    group = set(list(np.arange(num_task)[row_clusters == i]))
    group.update(list(np.arange(num_task)[col_clusters == i]))
    print(np.array(list(group)))
