import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_sparse import SparseTensor

from models.Precomputing_base import PrecomputingBase

class SIGN(PrecomputingBase):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, use_bn=True):
        super(SIGN, self).__init__(in_channels, hidden_channels, out_channels, num_layers,
                 dropout, use_bn)

        self.lins = torch.nn.ModuleList()
        for _ in range(self.num_layers + 1):
            self.lins.append(torch.nn.Linear(self.num_feats, self.dim_hidden))
        self.out_lin = torch.nn.Linear((self.num_layers + 1) * self.dim_hidden, self.num_classes)

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

        self.out_lin.reset_parameters()

    def forward(self, xs, return_softmax=True):
        outs = []
        for i, (x, lin) in enumerate(zip(xs, self.lins)):
            out = F.dropout(F.relu(lin(x)), p=self.dropout, training=self.training)
            outs.append(out)
        x = torch.cat(outs, dim=-1)
        x = self.out_lin(x)
        if return_softmax:
            x = F.log_softmax(x, dim=1)
        return x

class SIGN_MLP(PrecomputingBase):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, use_bn=True, mlp_layers=2, input_drop=0.3):
        super(SIGN_MLP, self).__init__(in_channels, hidden_channels, out_channels, num_layers,
                 dropout, use_bn)

        in_feats = self.num_feats
        out_feats = self.num_classes
        hidden = self.dim_hidden
        num_hops = self.num_layers + 1
        dropout = self.dropout

        self.dropout = nn.Dropout(dropout)
        self.prelu = nn.PReLU()
        self.inception_ffs = nn.ModuleList()
        self.input_drop = nn.Dropout(input_drop)
        for hop in range(num_hops):
            self.inception_ffs.append(
                FeedForwardNet(in_feats, hidden, hidden, mlp_layers, dropout))
        self.project = FeedForwardNet(num_hops * hidden, hidden, out_feats,
                                      mlp_layers, dropout)

    def forward(self, feats, return_softmax=True):
        feats = [self.input_drop(feat) for feat in feats]
        hidden = []
        for feat, ff in zip(feats, self.inception_ffs):
            hidden.append(ff(feat))
        out = self.project(self.dropout(self.prelu(torch.cat(hidden, dim=-1))))
        if return_softmax:
            out = F.log_softmax(out, dim=1)
        return out

    def reset_parameters(self):
        for ff in self.inception_ffs:
            ff.reset_parameters()
        self.project.reset_parameters()

class FeedForwardNet(nn.Module):
    def __init__(self, in_feats, hidden, out_feats, n_layers, dropout):
        super(FeedForwardNet, self).__init__()
        self.layers = nn.ModuleList()
        self.n_layers = n_layers
        if n_layers == 1:
            self.layers.append(nn.Linear(in_feats, out_feats))
        else:
            self.layers.append(nn.Linear(in_feats, hidden))
            for i in range(n_layers - 2):
                self.layers.append(nn.Linear(hidden, hidden))
            self.layers.append(nn.Linear(hidden, out_feats))
        if self.n_layers > 1:
            self.prelu = nn.PReLU()
            self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight, gain=gain)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        for layer_id, layer in enumerate(self.layers):
            x = layer(x)
            if layer_id < self.n_layers - 1:
                x = self.dropout(self.prelu(x))
        return x