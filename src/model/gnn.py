import torch
import torch.nn as nn
import torch.nn.functional as F


from torch_geometric.nn import GCNConv
from src.model.gnn_layer.gat_layer import GATConv
import src.model.gnn_wrapper as gnn_wrapper


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, num_heads=-1):
        super(GCN, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t, edge_attr=None):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x, edge_attr


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, num_heads=4):
        super(GAT, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads=num_heads, concat=False))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels, hidden_channels, heads=num_heads, concat=False))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GATConv(hidden_channels, out_channels, heads=num_heads, concat=False))
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index, edge_attr=None):
        for i, conv in enumerate(self.convs[:-1]):
            x, edge_attr = conv(x, edge_index=edge_index, edge_attr=edge_attr, )
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x, edge_attr = self.convs[-1](x, edge_index=edge_index, edge_attr=edge_attr)
        return x, edge_attr


class BioASQGAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, num_heads=4):
        super(BioASQGAT, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads=num_heads, concat=False))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels, hidden_channels, heads=num_heads, concat=False))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GATConv(hidden_channels, out_channels, heads=num_heads, concat=False))
        self.dropout = dropout
        self.edge_encoder = nn.Embedding(98*2, hidden_channels)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        e = self.edge_encoder(edge_attr)
        for i, conv in enumerate(self.convs[:-1]):
            x, e = conv(x, edge_index=edge_index, edge_attr=e)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x, e = self.convs[-1](x, edge_index=edge_index, edge_attr=e)
        return x, e


class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, res=True, model_name='GATConv'):
        super(GNN, self).__init__()
        self.dropout = dropout
        self.res = res

        self.node_encoder = nn.Linear(in_channels, hidden_channels)
        self.edge_encoder = nn.Embedding(98*2, hidden_channels)

        self.convs = nn.ModuleList(getattr(gnn_wrapper, model_name)(hidden_channels, hidden_channels) for _ in range(num_layers))
        self.bns = nn.ModuleList(torch.nn.BatchNorm1d(hidden_channels) for _ in range(num_layers))

        self.readout = nn.Sequential(
            nn.Linear(hidden_channels, out_channels//2),
            nn.Sigmoid(),
            nn.Linear(out_channels//2, out_channels))

    def forward(self, x, edge_index, edge_attr):
        x = self.node_encoder(x)
        e = self.edge_encoder(edge_attr)

        previous_x = x
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, e)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.res:
                x = x + previous_x
                previous_x = x

        x = self.readout(x)
        return x, e


load_gnn_model = {
    'gcn': GCN,
    'gat': GAT,
    'gnn': GNN,
    'bioasq_gat': BioASQGAT,
}
