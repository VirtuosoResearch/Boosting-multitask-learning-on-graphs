import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.nn import NNConv, Set2Set
from torch_geometric.data import DataLoader
from torch_geometric.utils import remove_self_loops


class MPNN(torch.nn.Module):
    def __init__(self,
                 node_input_dim=15,
                 edge_input_dim=5,
                 output_dim=1,
                 node_hidden_dim=64,
                 edge_hidden_dim=128,
                 num_step_message_passing=6,
                 num_step_set2set=6):
        super(MPNN, self).__init__()
        self.num_step_message_passing = num_step_message_passing
        self.lin0 = nn.Linear(node_input_dim, node_hidden_dim)
        edge_network = nn.Sequential(
            nn.Linear(edge_input_dim, edge_hidden_dim), nn.ReLU(),
            nn.Linear(edge_hidden_dim, node_hidden_dim * node_hidden_dim))
        self.conv = NNConv(node_hidden_dim,
                           node_hidden_dim,
                           edge_network,
                           aggr='mean',
                           root_weight=False)
        self.gru = nn.GRU(node_hidden_dim, node_hidden_dim)

        self.set2set = Set2Set(node_hidden_dim,
                               processing_steps=num_step_set2set)
        self.lin1 = nn.Linear(2 * node_hidden_dim, node_hidden_dim)
        self.lin2 = nn.Linear(node_hidden_dim, output_dim)
    
    def reset_parameters(self):
        self.lin0.reset_parameters()
        self.conv.reset_parameters()
        self.gru.reset_parameters()
        self.set2set.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        out = F.relu(self.lin0(data.x))
        h = out.unsqueeze(0)

        for i in range(self.num_step_message_passing):
            m = F.relu(self.conv(out, data.edge_index, data.edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)

        out = self.set2set(out, data.batch)
        out = F.relu(self.lin1(out))
        out = self.lin2(out)
        return out