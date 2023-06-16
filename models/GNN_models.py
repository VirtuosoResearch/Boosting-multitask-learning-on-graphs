import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
import torch_geometric.nn as geo_nn
from torch.nn import Linear
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
from torch_sparse import SparseTensor

from models.layers.reweighted_gcn_conv import ReweightedGCNConv, dispersion_norm
from models.layers.gat_conv import GATConv
from models.layers.gin_conv import GINConv
from models.layers.sage_conv import SAGEConv

class MLP(torch.nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(MLP, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.classifier = Linear(hidden_channels, out_channels)

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        self.classifier.reset_parameters()

    def forward(self, x, return_softmax = True, **kwargs):
        for _, lin in enumerate(self.lins):
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.classifier(x)

        if return_softmax == True:
            x = x.log_softmax(dim=-1)

        return x

    @torch.no_grad()
    def inference(self, x_all, device, return_softmax = True, **kwargs):
        x_all = x_all.to(device)
        for _, lin in enumerate(self.lins):
            x_all = lin(x_all)
            x_all = F.relu(x_all)
            x_all = F.dropout(x_all, p=self.dropout, training=self.training)
        x_all = self.classifier(x_all)
        if return_softmax:
            x_all = x_all.log_softmax(dim=-1)
        return x_all.cpu()

class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, use_bn=True):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.classifier = Linear(num_layers*hidden_channels, out_channels)

        self.dropout = dropout
        self.use_bn = use_bn

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.classifier.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None, return_softmax = True, **kwargs):
        xs = []
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_weight=edge_weight)
            if self.use_bn:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            xs.append(x)
        x = torch.cat(xs, dim=-1)
        x = self.classifier(x)

        if return_softmax == True:
            x = x.log_softmax(dim=-1)

        return x

    @torch.no_grad()
    def inference(self, x_all, test_loader, device, return_softmax = True):
        x_alls = []
        for i, conv in enumerate(self.convs):
            xs = []
            for batch_size, n_id, adj in test_loader:
                edge_index, _, size = adj
                edge_index = edge_index.to(device)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = conv((x, x_target), edge_index)
                if self.use_bn: 
                    x = self.bns[i](x)
                x = F.relu(x)
                xs.append(x.cpu())

            x_all = torch.cat(xs, dim=0)
            x_alls.append(x_all)
        x_all = torch.cat(x_alls, dim=1).to(device)
        x_all = self.classifier(x_all)
        if return_softmax:
            x_all = x_all.log_softmax(dim=-1)
        return x_all.cpu()

class IdentityModule(torch.nn.Module):
    """An identity module that outputs the input."""

    def __init__(self):
        super(IdentityModule, self).__init__()

    def forward(self, x): 
        return x

class GIN(torch.nn.Module):
    
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, 
                dropout, use_bn=True):
        super(GIN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.downsamples = torch.nn.ModuleList() # Downsample for skip connections
        self.bns = torch.nn.ModuleList()
        for _ in range(num_layers-1):
            mlp = geo_nn.MLP([in_channels, hidden_channels, hidden_channels], dropout=dropout, batch_norm=False)
            self.convs.append(GINConv(nn=mlp, train_eps=False))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
            if in_channels != hidden_channels:
                downsample = Linear(in_channels, hidden_channels, bias=False)
                self.downsamples.append(downsample)
            else:
                self.downsamples.append(IdentityModule())
            in_channels = hidden_channels
        mlp = geo_nn.MLP([hidden_channels, hidden_channels, out_channels], dropout=dropout, batch_norm=False)
        self.convs.append(GINConv(nn=mlp, train_eps=False))

        self.dropout = dropout
        self.use_bn = use_bn
        
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None, **kwargs):
        for i, conv in enumerate(self.convs[:-1]):
            out = conv(x, edge_index, edge_weight=edge_weight)
            if self.use_bn: 
                out = self.bns[i](out)
            x = F.relu(out) + self.downsamples[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index, edge_weight=edge_weight)

        return x.log_softmax(dim=-1)

    @torch.no_grad()
    def inference(self, x_all, test_loader, device, return_softmax = True):
        # x_alls = []
        for i, conv in enumerate(self.convs):
            xs = []
            for batch_size, n_id, adj in test_loader:
                edge_index, _, size = adj
                edge_index = edge_index.to(device)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                out = conv((x, x_target), edge_index)

                if i < len(self.convs)-1:
                    if self.use_bn: 
                        out = self.bns[i](out)
                    out = F.relu(out) + self.downsamples[i](x_target)
                else:
                    out = F.relu(out)
                xs.append(out.cpu())

            x_all = torch.cat(xs, dim=0)
            # x_alls.append(x_all)
        # x_all = torch.cat(x_alls, dim=1).to(device)
        # x_all = self.classifier(x_all)
        if return_softmax:
            x_all = x_all.log_softmax(dim=-1)
        return x_all.cpu()

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, use_bn=True, graph_pooling="mean"):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        
        self.classifier = Linear(num_layers*hidden_channels+in_channels, out_channels)
        self.dropout = dropout
        self.use_bn = use_bn

        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        else:
            self.pool = None
            print("No such pooling function!")
            
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.classifier.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None, batch=None, return_softmax = True):
        xs = [x]
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if self.use_bn:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            xs.append(x)
        x = torch.cat(xs, dim=-1)
        
        if (self.pool is not None) and (batch is not None):
            x = self.pool(x, batch)
        
        x = self.classifier(x)

        if return_softmax:
            x =  x.log_softmax(dim=-1)
        
        return x

'''Deprecated models'''


class ReweightedGCN(GCN):

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, graph_classification=False, normalize=True):
        super().__init__(in_channels, hidden_channels, out_channels, num_layers, dropout, graph_classification, normalize)
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(ReweightedGCNConv(in_channels, hidden_channels, normalize=normalize))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                ReweightedGCNConv(hidden_channels, hidden_channels, normalize=normalize))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        
        if graph_classification:
            self.convs.append(ReweightedGCNConv(hidden_channels, hidden_channels, normalize=normalize))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
            self.pred_head = Linear(hidden_channels, out_channels)
        else:
            self.convs.append(ReweightedGCNConv(hidden_channels, out_channels, normalize=normalize))

        self.dropout = dropout
        self.graph_classification = graph_classification

    def forward(self, x, adj_t, k_hop_nbrs, batch=None, use_bn=True, **kwargs):
        
        if self.graph_classification:
            for i, conv in enumerate(self.convs):
                x = conv(x, adj_t)
                if use_bn:
                    x = self.bns[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            # Apply a readout pooling
            assert batch is not None
            x = global_mean_pool(x, batch)
            # Apply a linear classifier
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.pred_head(x)
        else:
            for i, conv in enumerate(self.convs[:-1]):
                if isinstance(adj_t, SparseTensor):
                    adj_t = dispersion_norm(x, adj_t, k_hop_nbrs)
                    x = conv(x, adj_t)
                else:
                    edge_weight = dispersion_norm(x, adj_t, k_hop_nbrs)
                    x = conv(x, adj_t, edge_weight=edge_weight)
                if use_bn:
                    x = self.bns[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            if isinstance(adj_t, SparseTensor):
                adj_t = dispersion_norm(x, adj_t, k_hop_nbrs)
                x = self.convs[-1](x, adj_t)
            else:
                edge_weight = dispersion_norm(x, adj_t, k_hop_nbrs)
                x = self.convs[-1](x, adj_t, edge_weight=edge_weight)
        return x.log_softmax(dim=-1)

class ElementWiseLinear(torch.nn.Module):
    def __init__(self, size, weight=True, bias=True, inplace=False):
        super().__init__()
        if weight:
            self.weight = torch.nn.Parameter(torch.Tensor(1, size))
        else:
            self.weight = None
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(1, size))
        else:
            self.bias = None
        self.inplace = inplace

        self.reset_parameters()

    def reset_parameters(self):
        if self.weight is not None:
            torch.nn.init.ones_(self.weight)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(self, x):
        if self.inplace:
            if self.weight is not None:
                x.mul_(self.weight)
            if self.bias is not None:
                x.add_(self.bias)
        else:
            if self.weight is not None:
                x = x * self.weight
            if self.bias is not None:
                x = x + self.bias
        return x

class GAT(torch.nn.Module):
    
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
        num_heads, dropout, input_drop=0.0, attn_drop=0.0):
        super(GAT, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.num_heads = num_heads

        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads=num_heads, dropout=attn_drop))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels*num_heads))

        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(hidden_channels*num_heads, hidden_channels, heads=num_heads, dropout=attn_drop))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels*num_heads))
        self.convs.append(GATConv(hidden_channels*num_heads, out_channels, heads=1, dropout=attn_drop))
        
        self.bias_last = ElementWiseLinear(out_channels, weight=False, bias=True, inplace=True)
        self.input_drop = input_drop
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.bias_last.reset_parameters()

    def forward(self, x, adj_t, edge_weight=None, edge_attr=None, **kwargs):
        x = F.dropout(x, p=self.input_drop, training=self.training)
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t, edge_weight=edge_weight)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, adj_t, edge_weight=edge_weight)
        x = self.bias_last(x)
        return x.log_softmax(dim=-1)
