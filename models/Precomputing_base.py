import json
import os
import time

import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from torch_geometric.transforms import SIGN
from torch_sparse import SparseTensor


class PrecomputingBase(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, use_bn=True):
        super(PrecomputingBase, self).__init__()

        self.num_layers = num_layers
        self.dim_hidden = hidden_channels
        self.num_classes = out_channels
        self.num_feats = in_channels
        self.dropout =dropout
        self.use_bn = use_bn

    def forward(self, xs, return_softmax=True):
        raise NotImplementedError

    @torch.no_grad()
    def inference(self, xs_all, device, return_softmax=True):
        y_preds = []
        loader = DataLoader(range(xs_all[0].size(0)), batch_size=100000)
        for perm in loader:
            y_pred = self.forward([x[perm].to(device) for x in xs_all], return_softmax=return_softmax)
            y_preds.append(y_pred.cpu())
        y_preds = torch.cat(y_preds, dim=0)

        return y_preds
