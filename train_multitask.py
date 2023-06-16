import argparse
import os.path as osp

import os
import numpy as np
import time
import torch
import torch.nn.functional as F
import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.datasets import TUDataset, QM9
from torch_geometric.loader import DataLoader
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
from torch_geometric.utils import to_scipy_sparse_matrix, degree, to_undirected, remove_self_loops
from torch_geometric.data import Data
from torch_geometric.loader import NeighborSampler
from utils.loader import OriginalGraphDataLoader, GraphSAINTNodeSampler, GraphSAINTEdgeSampler, GraphSAINTRandomWalkSampler, LeverageScoreEdgeSampler

from ogb.graphproppred import PygGraphPropPredDataset
from models import *
from utils.pagerank import pagerank_scipy
from utils.util import add_result_to_csv, precompute, generate_masked_labels
from minibatch_trainer import Trainer, MultitaskTrainer, RegressionTrainer

name_to_samplers = {
    "no_sampler": OriginalGraphDataLoader,
    "node_sampler": GraphSAINTNodeSampler,
    "edge_sampler": GraphSAINTEdgeSampler,
    "rw_sampler": GraphSAINTRandomWalkSampler,
    "ls_sampler": LeverageScoreEdgeSampler
}

name_to_num_classes = {
    "youtube": 100,
    "dblp": 100,
    "amazon": 100,
    "livejournal": 100,
    "alchemy_full": 12,
    "QM9": 12,
    "molpcba": 128
}


def split_dataset(dataset, train_ratio, val_ratio, train_size):
    dataset_length = len(dataset)
    rng = np.random.default_rng(42)
    permutations = rng.permutation(dataset_length)
    train_idx = permutations[:int(train_ratio*dataset_length)]
    val_idx = permutations[int(train_ratio*dataset_length):int((train_ratio+val_ratio)*dataset_length)]
    test_idx = permutations[int((train_ratio+val_ratio)*dataset_length):]

    train_idx = train_idx[:train_size]

    train_dataset = dataset.copy(torch.tensor(train_idx))
    valid_dataset = dataset.copy(torch.tensor(val_idx)) 
    test_dataset = dataset.copy(torch.tensor(test_idx))
    return train_dataset, valid_dataset, test_dataset

class Complete(object):
    def __call__(self, data):
        device = data.edge_index.device

        row = torch.arange(data.num_nodes, dtype=torch.long, device=device)
        col = torch.arange(data.num_nodes, dtype=torch.long, device=device)

        row = row.view(-1, 1).repeat(1, data.num_nodes).view(-1)
        col = col.repeat(data.num_nodes)
        edge_index = torch.stack([row, col], dim=0)

        edge_attr = None
        if data.edge_attr is not None:
            idx = data.edge_index[0] * data.num_nodes + data.edge_index[1]
            size = list(data.edge_attr.size())
            size[0] = data.num_nodes * data.num_nodes
            edge_attr = data.edge_attr.new_zeros(size)
            edge_attr[idx] = data.edge_attr

        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
        data.edge_attr = edge_attr
        data.edge_index = edge_index

        return data

def main(args):
    start = time.time()
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    if args.dataset == 'alchemy_full':
        dataset = TUDataset('./data/TUDataset', name="alchemy_full")
        rng = np.random.default_rng(42)
        permutations = rng.permutation(len(dataset))
        dataset = dataset[permutations]

        mean = dataset.data.y.mean(dim=0, keepdim=True)
        std = dataset.data.y.std(dim=0, keepdim=True)
        dataset.data.y = (dataset.data.y - mean) / std
        mean, std = mean.to(device), std.to(device)

        train_dataset = dataset[:162063].shuffle()
        if args.downsample < 1:
            train_dataset = train_dataset[:int(args.downsample * len(train_dataset))]
        valid_dataset = dataset[162063:182321].shuffle()
        test_dataset = dataset[182321:].shuffle()

        edge_features = 4
        node_features = 6
    elif args.dataset == 'QM9':
        dataset = QM9('./data/TUDataset/QM9', transform=T.Compose([Complete(), T.Distance(norm=False)]))
        dataset.data.y = dataset.data.y[:, 0:12]

        rng = np.random.default_rng(42)
        permutations = rng.permutation(len(dataset))
        dataset = dataset[permutations]

        tenpercent = int(len(dataset) * 0.1)
        mean = dataset.data.y.mean(dim=0, keepdim=True)
        std = dataset.data.y.std(dim=0, keepdim=True)
        dataset.data.y = (dataset.data.y - mean) / std
        mean, std = mean.to(device), std.to(device)

        tenpercent = int(len(dataset) * 0.1)
        test_dataset = dataset[:tenpercent].shuffle()
        valid_dataset = dataset[tenpercent:2 * tenpercent].shuffle()
        train_dataset = dataset[2 * tenpercent:].shuffle()
        if args.downsample < 1:
            train_dataset = train_dataset[:int(args.downsample * len(train_dataset))]

        edge_features = 5
        node_features = 11
    elif args.dataset == 'molpcba':
        dataset = PygGraphPropPredDataset(name = "ogbg-molpcba")
        split_idx = dataset.get_idx_split()
        test_dataset = dataset[split_idx["test"]]
        valid_dataset = dataset[split_idx["valid"]]
        train_dataset = dataset[split_idx["train"]]
        if args.downsample < 1:
            train_dataset = train_dataset[:int(args.downsample * len(train_dataset))]
    else:
        print("Non-valid dataset name!")
        exit()
    num_classes = name_to_num_classes[args.dataset]

    if args.task_idxes == -1:
        task_idxes = np.arange(num_classes)
    else:
        task_idxes = np.array(args.task_idxes)
    
    # train_dataset, valid_dataset, test_dataset = split_dataset(dataset, args.train_ratio, args.val_ratio, args.train_size)
    print("Training size: {} Validation size: {} Test size: {}".format(
        len(train_dataset), len(valid_dataset), len(test_dataset)
    ))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size*4, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size*4, shuffle=False)

    # Initialize the model
    assert args.model == "gine"
    if args.dataset == 'alchemy_full':
        model = NetGINE_v2(edge_features, node_features, dim=args.hidden_channels, num_classes=num_classes)
    elif args.dataset == 'QM9':
        model = NetGINE(edge_features, node_features, dim=args.hidden_channels, num_classes=num_classes)
    elif args.dataset == "molpcba":
        model = GNN_MOL(
            gnn_type = "gin",
            num_tasks=num_classes,
            num_layer=5,
            emb_dim=args.hidden_channels,
            virtual_node = True,
            drop_ratio=0.5
        )
    print(model)
    model = model.to(device)

    log_metrics = {}
    for run in range(args.runs):
        # reintialize model and optimizer
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                        factor=0.5, patience=5,
                                                        min_lr=0.0000001)

        task_idxes_str = str(args.task_idxes) if args.task_idxes == -1 else "_".join([str(idx) for idx in args.task_idxes])
        if len(task_idxes_str) > 100:
            task_idxes_str = task_idxes_str[:100]
        trainer = RegressionTrainer(model, optimizer, dataset[0], train_loader, valid_loader, test_loader, device,
                        epochs=args.epochs, log_steps=args.log_steps, degrees=None, degree_thres=0, 
                        criterion = args.criterion, evaluator = args.evaluator, monitor=args.monitor, decoupling=False,
                        checkpoint_dir=f"./saved/{args.dataset}_{args.model}_{args.hidden_channels}_{task_idxes_str}",
                        task_idxes=task_idxes, lr_scheduler=lr_scheduler, 
                        mnt_mode=args.mnt_mode,
                        eval_separate=args.eval_separate)
        _ = trainer.train()
        trainer.load_checkpoint()
        log = trainer.test()
        
        for key, val in log.items():
            if key in log_metrics:
                log_metrics[key].append(val)
            else:
                log_metrics[key] = [val, ]
    print("Test {}: {:.4f}±{:.4f}".format(
            args.evaluator,
            np.mean(log_metrics[f"test_{args.evaluator}"]), 
            np.std(log_metrics[f"test_{args.evaluator}"])
        ))
    
    # save results into .csv
    file_dir = os.path.join("./results/", args.save_name)
    if not os.path.exists(file_dir):
        os.mkdir(file_dir)
    
    for task_idx in task_idxes:
        # save validation results
        result_datapoint = {
            "Task": task_idx, 
            "Trained on": task_idxes,
        }
        for key, vals in log_metrics.items():
            if f"task_{task_idx}" in key:
                metric_name = "_".join(key.split("_")[2:])
                result_datapoint[metric_name] = np.mean(vals)
                result_datapoint[metric_name+"_std"] = np.std(vals)
        file_name = os.path.join(file_dir, "{}_{}.csv".format(args.save_name, args.dataset))
        add_result_to_csv(result_datapoint, file_name)
    end = time.time()
    print("Training completes in {} seconds".format(end-start))


def main_v2(args):
    start = time.time()
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    if args.dataset == "youtube" or args.dataset == "dblp" or args.dataset == "livejournal" or args.dataset == "amazon":
        transform = T.ToUndirected()
        data_dir = f"./data/com_{args.dataset}/"
        assert os.path.exists(os.path.join(data_dir, f'{args.dataset}_{args.num_communities}_{args.feature_dim}_data.pt'))
        data = torch.load(os.path.join(data_dir, f'{args.dataset}_{args.num_communities}_{args.feature_dim}_data.pt'))
        print("Load data from file!")
    else:
        print("Non-valid dataset name!")
        exit()
    num_classes = name_to_num_classes[args.dataset]

    if args.task_idxes == -1:
        task_idxes = np.arange(num_classes)
    else:
        task_idxes = np.array(args.task_idxes)
    
    data.y = data.y[:, task_idxes]
    if len(data.train_mask.shape) == 2:
        data.train_mask = data.train_mask[:, task_idxes]
        data.val_mask  = data.val_mask[:, task_idxes]
        data.test_mask = data.test_mask[:, task_idxes]

    ''' Downsample training set'''
    if args.downsample < 1.0:
        if len(data.train_mask.shape) == 2:
            for idx in range(data.train_mask.shape[1]):
                masked_labels = generate_masked_labels(data.train_mask[:, idx], args.downsample)
                data.train_mask[:, idx][masked_labels] = False
            print("Training size: {}".format(data.train_mask[:, 0].sum().item()))
        else:
            masked_labels = generate_masked_labels(data.train_mask, args.downsample)
            data.train_mask[masked_labels] = False
            print("Training size: {}".format(data.train_mask.sum().item()))
        args.batch_size = int(args.batch_size/args.downsample)

    degrees = None; degree_thres = 0
    
    # Initialize mini-batch sampler
    decoupling = args.sample_method=="decoupling"
    if decoupling:
        data = precompute(data, args.num_layers)

        xs_train = torch.cat([x for x in data.xs], -1)
        y_train = data.y

        train_set = torch.utils.data.TensorDataset(xs_train, y_train, data.train_mask) \
            if not (args.model == "dmon" or args.model == "mincut")  else torch.utils.data.TensorDataset(xs_train, y_train, data.train_mask, torch.arange(xs_train.shape[0]))
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=args.batch_size, num_workers=1
        )
        test_loader = None
    else:
        Z_dir = f"./save_z/{args.dataset}_z.npy"
        train_loader = name_to_samplers[args.sample_method](data, batch_size=args.batch_size,
                                    num_steps=args.num_steps, sample_coverage=args.sample_coverage,
                                    walk_length=args.walk_length, Z_dir=Z_dir)

        test_loader = NeighborSampler(data.clone().edge_index, sizes=[-1],
                                        batch_size=args.test_batch_size, shuffle=False,
                                        num_workers=1)

    # Initialize the model
    if args.model == "mlp":
        model = MLP(data.num_features, args.hidden_channels,
                     len(task_idxes), args.num_layers,
                     args.dropout)
    elif args.model == "sign":
        model = SIGN_MLP(data.num_features, args.hidden_channels,
                     len(task_idxes), args.num_layers,
                     args.dropout, use_bn=not args.no_bn, 
                     mlp_layers=args.mlp_layers, input_drop=args.input_drop)
    elif args.model == "gamlp":
        model = JK_GAMLP(data.num_features, args.hidden_channels,
                     len(task_idxes), args.num_layers,
                     args.dropout, use_bn=not args.no_bn, 
                     input_drop=args.input_drop, att_dropout=args.attn_drop, pre_process=True, residual=True, alpha=args.alpha)
    elif args.model == "moe":
        model = MixtureOfExperts(data.num_features, args.hidden_channels,
                     len(task_idxes), args.num_layers,
                     args.dropout, use_bn=not args.no_bn,
                     num_of_experts = args.num_of_experts)
    else:
        raise NotImplementedError("No such model implementation!")
    print(model)
    model = model.to(device)

    log_metrics = {}
    for run in range(args.runs):
        # reintialize model and optimizer
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        
        task_idxes_str = str(args.task_idxes) if args.task_idxes == -1 else "_".join([str(idx) for idx in args.task_idxes])
        if len(task_idxes_str) > 100:
            task_idxes_str = task_idxes_str[:100]
        
        trainer = MultitaskTrainer(model, optimizer, data, train_loader, test_loader, device,
                        epochs=args.epochs, log_steps=args.log_steps, degrees=degrees, degree_thres=degree_thres, 
                        criterion = "multilabel", evaluator = args.evaluator, monitor=args.monitor, decoupling=decoupling,
                        checkpoint_dir=f"./saved/{args.dataset}_{args.model}_{args.num_layers}_{args.hidden_channels}_{task_idxes_str}",
                        task_idxes=task_idxes)
        _, _ = trainer.train()
        trainer.load_checkpoint()
        
        if len(data.train_mask.shape) == 2:
            log = trainer.test_in_task_mask()
        else:
            log = trainer.test()
        
        for key, val in log.items():
            if key in log_metrics:
                log_metrics[key].append(val)
            else:
                log_metrics[key] = [val, ]
    print("Test accuracy: {:.4f}±{:.4f}".format(
            np.mean(log_metrics[f"test_{args.evaluator}"]), 
            np.std(log_metrics[f"test_{args.evaluator}"])
        ))
    print("Test accuracy for degree <={:2.0f}: {:.4f}±{:.4f}".format(
            degree_thres, 
            np.mean(log_metrics[f"test_longtail_{args.evaluator}"]), 
            np.std(log_metrics[f"test_longtail_{args.evaluator}"])
        ))
    
    # save results into .csv
    file_dir = os.path.join("./results/", args.save_name)
    if not os.path.exists(file_dir):
        os.mkdir(file_dir)
    
    for task_idx in task_idxes:
        # save validation results
        result_datapoint = {
            "Task": task_idx, 
            "Trained on": task_idxes,
        }
        for key, vals in log_metrics.items():
            if f"task_{task_idx}" in key:
                metric_name = "_".join(key.split("_")[2:])
                result_datapoint[metric_name] = np.mean(vals)
                result_datapoint[metric_name+"_std"] = np.std(vals)
        file_name = os.path.join(file_dir, "{}_{}.csv".format(args.save_name, args.dataset))
        add_result_to_csv(result_datapoint, file_name)
    end = time.time()
    print("Training completes in {} seconds".format(end-start))

def add_decoupling_args(parser):
    # For SIGN
    parser.add_argument("--mlp_layers", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=0.5)

    # For MOE
    parser.add_argument('--num_of_experts', type=int, default=10)

    # For DMoN
    parser.add_argument('--dmon_lam', type=float, default=1.0)
    return parser

def add_community_detection_args(parser):
    # num_cmty=100, train_ratio=0.2, val_ratio=0.2, feature_dim=64
    parser.add_argument("--num_communities", type=int, default=100)
    parser.add_argument("--train_ratio", type=float, default=0.2)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--feature_dim", type=int, default=128)
    return parser

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='youtube')
    parser.add_argument('--model', type=str, default='sign')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--use_edge_index', action="store_true")
    parser.add_argument('--criterion', type=str, default="regression")
    parser.add_argument('--evaluator', type=str, default="f1_score")
    parser.add_argument('--monitor', type=str, default="avg")
    parser.add_argument('--task_idxes', nargs='+', type=int, default=-1)
    parser.add_argument("--save_name", type=str, default="test")

    parser.add_argument('--mnt_mode', type=str, default="min")
    parser.add_argument('--eval_separate', action="store_true")

    ''' Sampling '''
    parser.add_argument('--sample_method', type=str, default="decoupling")
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--test_batch_size', type=int, default=20000)
    parser.add_argument('--walk_length', type=int, default=2)
    parser.add_argument('--num_steps', type=int, default=10)
    parser.add_argument('--sample_coverage', type=int, default=0)

    ''' Model '''
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--no_bn', action="store_true")
    
    # GAT
    parser.add_argument('--num_heads', type=int, default=3)
    parser.add_argument('--input_drop', type=float, default=0.3)
    parser.add_argument('--attn_drop', type=float, default=0.1)

    ''' Training '''
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--downsample', type=float, default=1.0)

    parser = add_decoupling_args(parser)
    parser = add_community_detection_args(parser)
    args = parser.parse_args()
    print(args)

    if args.dataset == "youtube" or args.dataset == "dblp" or args.dataset == "livejournal" or args.dataset == "amazon":
        main_v2(args)
    else:
        main(args)
