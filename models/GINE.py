from torch.nn import Linear as Lin
from torch.nn import Sequential, Linear, ReLU

import torch
from torch_geometric.nn import MessagePassing, Set2Set
import torch.nn.functional as F

class GINConv(MessagePassing):
    def __init__(self, emb_dim, dim1, dim2):
        super(GINConv, self).__init__(aggr="add")

        self.bond_encoder = Sequential(Linear(emb_dim, dim1), torch.nn.BatchNorm1d(dim1), ReLU(), Linear(dim1, dim1),
                                       torch.nn.BatchNorm1d(dim1), ReLU())

        self.mlp = Sequential(Linear(dim1, dim2), torch.nn.BatchNorm1d(dim2), ReLU(), Linear(dim2, dim2),
                              torch.nn.BatchNorm1d(dim2), ReLU())

        self.eps = torch.nn.Parameter(torch.Tensor([0]))

    def reset_parameters(self):
        for module in self.bond_encoder:
            if module.__class__.__name__ == 'ReLU':
                continue
            module.reset_parameters()
        for module in self.mlp:
            if module.__class__.__name__ == 'ReLU':
                continue
            module.reset_parameters()
        self.eps.data.fill_(0)

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.bond_encoder(edge_attr)

        out = self.mlp((1 + self.eps) * x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))

        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


class NetGINE(torch.nn.Module):
    def __init__(self, edge_features, node_features, dim, num_classes=12):
        super(NetGINE, self).__init__()

        num_features = node_features
        dim = dim

        self.conv1 = GINConv(edge_features, num_features, dim)
        self.conv2 = GINConv(edge_features, dim, dim)
        self.conv3 = GINConv(edge_features, dim, dim)
        self.conv4 = GINConv(edge_features, dim, dim)
        self.conv5 = GINConv(edge_features, dim, dim)
        self.conv6 = GINConv(edge_features, dim, dim)

        self.set2set = Set2Set(1 * dim, processing_steps=6)

        self.fc1 = Lin(2 * dim, dim)
        self.fc4 = Linear(dim, num_classes)

    def reset_parameters(self):
        for conv in [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6]:
            conv.reset_parameters()
        self.set2set.reset_parameters()
        self.fc1.reset_parameters()
        self.fc4.reset_parameters()

    def forward(self, data):
        x = data.x

        x_1 = F.relu(self.conv1(x, data.edge_index, data.edge_attr))
        x_2 = F.relu(self.conv2(x_1, data.edge_index, data.edge_attr))
        x_3 = F.relu(self.conv3(x_2, data.edge_index, data.edge_attr))
        x_4 = F.relu(self.conv4(x_3, data.edge_index, data.edge_attr))
        x_5 = F.relu(self.conv5(x_4, data.edge_index, data.edge_attr))
        x_6 = F.relu(self.conv6(x_5, data.edge_index, data.edge_attr))
        x = x_6

        x = self.set2set(x, data.batch)

        x = F.relu(self.fc1(x))
        x = self.fc4(x)
        return x
    


class GINConv_v2(MessagePassing):
    def __init__(self, emb_dim, dim1, dim2):
        super(GINConv_v2, self).__init__(aggr="add")

        self.bond_encoder = Sequential(Linear(emb_dim, dim1), ReLU(), Linear(dim1, dim1))
        self.mlp = Sequential(Linear(dim1, dim1), ReLU(), Linear(dim1, dim2))

        self.eps = torch.nn.Parameter(torch.Tensor([0]))
    
    def reset_parameters(self):
        for module in self.bond_encoder:
            if module.__class__.__name__ == 'ReLU':
                continue
            module.reset_parameters()
        for module in self.mlp:
            if module.__class__.__name__ == 'ReLU':
                continue
            module.reset_parameters()
        self.eps.data.fill_(0)

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.bond_encoder(edge_attr)

        out = self.mlp((1 + self.eps) * x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))

        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


class NetGINE_v2(torch.nn.Module):
    def __init__(self, edge_features, node_features, dim, num_classes=12):
        super(NetGINE_v2, self).__init__()

        num_features = node_features
        dim = dim

        self.conv1 = GINConv_v2(edge_features, num_features, dim)
        self.conv2 = GINConv_v2(edge_features, dim, dim)
        self.conv3 = GINConv_v2(edge_features, dim, dim)
        self.conv4 = GINConv_v2(edge_features, dim, dim)
        self.conv5 = GINConv_v2(edge_features, dim, dim)
        self.conv6 = GINConv_v2(edge_features, dim, dim)

        self.set2set = Set2Set(1 * dim, processing_steps=6)

        self.fc1 = Lin(2 * dim, dim)
        self.fc4 = Linear(dim, num_classes)

    def reset_parameters(self):
        for conv in [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6]:
            conv.reset_parameters()
        self.set2set.reset_parameters()
        self.fc1.reset_parameters()
        self.fc4.reset_parameters()

    def forward(self, data):
        x = data.x

        x_1 = F.relu(self.conv1(x, data.edge_index, data.edge_attr))
        x_2 = F.relu(self.conv2(x_1, data.edge_index, data.edge_attr))
        x_3 = F.relu(self.conv3(x_2, data.edge_index, data.edge_attr))
        x_4 = F.relu(self.conv4(x_3, data.edge_index, data.edge_attr))
        x_5 = F.relu(self.conv5(x_4, data.edge_index, data.edge_attr))
        x_6 = F.relu(self.conv6(x_5, data.edge_index, data.edge_attr))
        x = x_6
        x = self.set2set(x, data.batch)
        x = F.relu(self.fc1(x))
        x = self.fc4(x)
        return x
