import os
import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
import torch_geometric
from sklearn.metrics import roc_auc_score, f1_score
from torch_geometric.transforms import SIGN
import time

def generate_masked_labels(train_mask, downsample):
    rng = np.random.default_rng(1024)
    valid_labels = torch.nonzero(train_mask, as_tuple=True)[0]
    valid_labels = valid_labels.numpy()
    masked_labels = rng.choice(valid_labels, size=int((1-downsample)*valid_labels.shape[0]), replace=False)
    return masked_labels

def precompute(data, num_layers): # processed_dir
    print("precomputing features, may take a while.")
    t1 = time.time()
    data = SIGN(num_layers)(data)
    data.xs = [data.x] + [data[f"x{i}"] for i in range(1, num_layers + 1)]
    t2 = time.time()
    print("precomputing finished using %.4f s." % (t2 - t1))
    return data

def add_result_to_csv(result_datapoint, file_name):
    for key, val in result_datapoint.items():
        result_datapoint[key] = [val, ]
    
    if os.path.exists(file_name):
        result_df = pd.read_csv(file_name, index_col=0)
        tmp_df = pd.DataFrame(result_datapoint)
        result_df = pd.concat([result_df, tmp_df], ignore_index = True)
        result_df.to_csv(file_name)
    else:
        result_df = pd.DataFrame(result_datapoint)  
        result_df.to_csv(file_name)   


def k_hop_neighbors(edge_index, num_nodes, num_hops):
    '''
    Find the k-hop neighbors of node indexed from 1 to num_nodes
        Assumes that flow == 'target_to_source'
    '''
    row, col = edge_index
    k_hop_nbrs = []
    for idx in range(num_nodes):
        node_mask = row.new_empty(num_nodes, dtype=torch.bool)
        edge_mask = row.new_empty(row.size(0), dtype=torch.bool)
        
        node_idx = torch.tensor([idx], device=row.device).flatten()
        subsets = [node_idx]

        for _ in range(num_hops):
            node_mask.fill_(False)
            node_mask[subsets[-1]] = True
            torch.index_select(node_mask, 0, row, out=edge_mask)
            subsets.append(col[edge_mask])

        subset = torch.cat(subsets).unique()
        k_hop_nbrs.append(subset)
    return k_hop_nbrs

development_seed = 1684992425

def set_train_val_test_split(seed, data, num_development = 1500, num_per_class = 20):
    rnd_state = np.random.RandomState(development_seed)
    num_nodes = data.y.shape[0]
    development_idx = rnd_state.choice(num_nodes, num_development, replace=False)
    test_idx = [i for i in np.arange(num_nodes) if i not in development_idx]

    train_idx = []
    rnd_state = np.random.RandomState(seed)
    for c in range(data.y.max() + 1):
        class_idx = development_idx[np.where(data.y[development_idx].cpu() == c)[0]]
        train_idx.extend(rnd_state.choice(class_idx, num_per_class, replace=False))

    val_idx = [i for i in development_idx if i not in train_idx]

    def get_mask(idx):
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        mask[idx] = 1
        return mask

    data.train_mask = get_mask(train_idx)
    data.val_mask = get_mask(val_idx)
    data.test_mask = get_mask(test_idx)

    return data