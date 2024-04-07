import torch
import torch.nn.functional as F

from torch_geometric.nn import GCN, GAE,GAT
from torch_geometric.nn import GCNConv,GATConv,GATv2Conv
import torch.nn as nn

class GCN_NN(torch.nn.Module):
    def __init__(self, feature_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GCNConv(feature_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)

    def forward(self, x, edge_index, edge_weight=None):
        h = F.relu(self.conv1(x, edge_index, edge_weight=edge_weight))
        logits = self.conv2(h, edge_index, edge_weight=edge_weight)
        return logits

class GAT_NN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads, num_layers):
        super(GAT_NN, self).__init__()
        self.gat = GAT(input_dim, hidden_dim,out_channels=output_dim, heads=num_heads, num_layers=num_layers)

    def forward(self, x, edge_index, edge_weight=None):
        return self.gat(x, edge_index,edge_attr=edge_weight)

class GAE_NN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
            super(GAE_NN, self).__init__()
            self.encoder = GCNConv(input_dim, hidden_dim)
            self.decoder = GCNConv(hidden_dim, input_dim)
            self.gae= GAE(self.encoder,self.decoder)

    def forward(self, x, edge_index, edge_weight=None):
            z = self.encoder(x, edge_index,edge_weight=edge_weight).relu()
            recon_x = self.decoder(z, edge_index)
            return z,recon_x

class LogisticRegression(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))
