import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

from torch.utils.data import DataLoader

import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(LinkPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, feature_dim=2):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(
            GCNConv(in_channels, hidden_channels, normalize=False))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, normalize=False))
        self.convs.append(
            GCNConv(hidden_channels, out_channels, normalize=False))

        self.dropout = dropout
        self.bceloss = nn.BCELoss()

        self.predictor = LinkPredictor(in_channels=out_channels,
                                       hidden_channels=hidden_channels,
                                       out_channels=feature_dim,
                                       num_layers=num_layers,
                                        dropout=dropout)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t, train_edges, x2=None, adj2=None, batch_num_nodes=None, **kwargs):

        x = torch.squeeze(x)
        adj_t = torch.squeeze(adj_t)

        src_idx = train_edges[:, 0]
        dst_idx = train_edges[:, 1]

        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        self.embedding_tensor = self.convs[-1](x, adj_t)

        if x2 is not None and adj2 is None:
            x2 = torch.squeeze(x2)
            adj2 = torch.squeeze(adj2)
            for conv in self.convs[:-1]:
                x2 = conv(x2, adj2)
                x2 = F.relu(x2)
                x2 = F.dropout(x2, p=self.dropout, training=self.training)
            self.dst_embedding_tensor = self.convs[-1](x2, adj2)

        if x2 is not None and adj2 is None:
            pred = self.pred_model(
                self.embedding_tensor[train_edges[0]], self.dst_embedding_tensor[train_edges[1]]
            )
        else:
            pred = self.predictor(self.embedding_tensor[src_idx], self.embedding_tensor[dst_idx])

        pred = torch.unsqueeze(pred, 0)
        return pred



    def loss(self, pred, label):
        #label = torch.squeeze(label)
        #pred = torch.squeeze(pred, -1)
        return self.bceloss(pred, label.float())