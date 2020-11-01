import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.nn import ModuleList, Linear, Conv1d, MaxPool1d, Embedding

from torch.utils.data import DataLoader

import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv

import numpy as np

import scipy.sparse as sp

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
        x = F.normalize(x, p=2, dim=-1)
        self.x = x
        self.x.retain_grad()
        return x.log_softmax(-1)

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                args, dropout=0, feature_dim=2):
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
        self.celoss = nn.CrossEntropyLoss()

        self.predictor = LinkPredictor(in_channels=out_channels,
                                       hidden_channels=hidden_channels,
                                       out_channels=feature_dim,
                                       num_layers=num_layers,
                                        dropout=dropout)

        self.args = args

        self.single_edge_label = args.single_edge_label
        self.multi_label = args.multi_label
        self.multi_class = args.multi_class

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.predictor.reset_parameters();

    def forward(self, x, adj_t, train_edges, x2=None, adj2=None, batch_num_nodes=None, **kwargs):
        adj_t_origin = adj_t
        adj2_origin = adj2

        src_edge_weight = None
        dst_edge_weight = None

        if type(adj_t) == sp.coo.coo_matrix:
            #adj_t = adj_t.tocoo().astype(np.float32)
            adj_t = torch.from_numpy(
                np.vstack((adj_t.row, adj_t.col)).astype(np.int64)
            )
        elif type(adj_t) == torch.Tensor:
            #adj_t = adj_t.tocoo().astype(np.float32)
            adj_t = torch.squeeze(adj_t)
            adj_t = adj_t + adj_t.transpose(0, 1)
            adj_t = torch.nonzero(adj_t).transpose(0,1)
        else:
            adj_t = adj_t[:, [0, 1]].transpose(0, 1)
            adj_t = torch.squeeze(adj_t)

        if adj2 is not None and type(adj2) == sp.coo.coo_matrix:
            adj2 = torch.from_numpy(
                np.vstack((adj2.row, adj2.col)).astype(np.int64)
            )
        elif adj2 is not None and type(adj2) == torch.Tensor:
            # adj_t = adj_t.tocoo().astype(np.float32)
            adj2 = torch.squeeze(adj2)
            adj2 = adj2 + adj2.transpose(0,1)
            adj2 = torch.nonzero(adj2).transpose(0, 1)
        elif adj2 is not None:
            adj2 = adj2[:, [0, 1]].transpose(0, 1)
            adj2 = torch.squeeze(adj2)

        if self.args.gpu:
            adj_t = adj_t.cuda()
            if adj2 is not None:
                adj2 = adj2.cuda()

        x = torch.squeeze(x)

        #adj_t = torch.cat((adj_t, adj_t[[1, 0], :]), -1)
        if adj2 is None:
            src_idx = train_edges[:, 0]
            dst_idx = train_edges[:, 1]
        else:
            src_idx = train_edges[0]
            dst_idx = train_edges[1]
            src_edge_weight = [adj_t_origin[0,i,j].unsqueeze(0) if i < j else adj_t_origin[0,j,i].unsqueeze(0) for [i, j] in adj_t.permute(1, 0)]
            src_edge_weight = torch.cat(src_edge_weight)
            self.src_edge_weight = src_edge_weight

            self.src_edge_weight.retain_grad()
            dst_edge_weight = [adj2_origin[0,i,j].unsqueeze(0) if i < j else adj2_origin[0,j,i].unsqueeze(0) for [i, j] in adj2.permute(1, 0)]
            dst_edge_weight = torch.cat(dst_edge_weight)
            self.dst_edge_weight = dst_edge_weight

            self.dst_edge_weight.retain_grad()

        for conv in self.convs[:-1]:
            x = conv(x, adj_t, edge_weight=src_edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        self.embedding_tensor = self.convs[-1](x, adj_t)
        #self.embedding_tensor = F.normalize(self.embedding_tensor, p=2, dim=-1)
        self.embedding_tensor.retain_grad()

        if x2 is not None and adj2 is not None:
            x2 = torch.squeeze(x2)
            adj2 = torch.squeeze(adj2)
            for conv in self.convs[:-1]:
                x2 = conv(x2, adj2, dst_edge_weight)
                x2 = F.relu(x2)
                x2 = F.dropout(x2, p=self.dropout, training=self.training)
            self.dst_embedding_tensor = self.convs[-1](x2, adj2)
            #self.dst_embedding_tensor = F.normalize(self.dst_embedding_tensor, p=2, dim=-1)
            self.dst_embedding_tensor.retain_grad()

        if x2 is not None and adj2 is not None:
            pred = self.predictor(
                self.embedding_tensor[train_edges[0]], self.dst_embedding_tensor[train_edges[1]]
            )
        else:
            pred = self.predictor(self.embedding_tensor[src_idx], self.embedding_tensor[dst_idx])

        pred = torch.unsqueeze(pred, 0)
        return pred



    def loss(self, pred, label):
        #label = torch.squeeze(label)
        #pred = torch.squeeze(pred, -1)
        if self.single_edge_label or self.multi_class:
            pred = torch.transpose(pred, 1, 2)
            return self.celoss(pred, label)
        elif self.multi_label:
            return self.bceloss(pred, label.float())

#from ogb_seal
class SAGE(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers, max_z,
                 use_feature=False, node_embedding=None,
                 dropout=0.5, num_features=3):
        super(SAGE, self).__init__()
        self.use_feature = use_feature
        self.node_embedding = node_embedding
        self.max_z = max_z
        self.z_embedding = Embedding(self.max_z, hidden_channels)

        self.convs = torch.nn.ModuleList()
        initial_channels = hidden_channels
        if self.use_feature:
            initial_channels += num_features
        if self.node_embedding is not None:
            initial_channels += node_embedding.embedding_dim
        self.convs.append(SAGEConv(initial_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))

        self.dropout = dropout
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, z, edge_index, batch, x=None, edge_weight=None, node_id=None):
        z_emb = self.z_embedding(z)
        if self.use_feature and x is not None:
            x = torch.cat([z_emb, x.to(torch.float)], 1)
        else:
            x = z_emb
        if self.node_embedding is not None and node_id is not None:
            n_emb = self.node_embedding(node_id)
            x = torch.cat([x, n_emb], 1)
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        if True:  # center pooling
            _, center_indices = np.unique(batch.cpu().numpy(), return_index=True)
            x_src = x[center_indices]
            x_dst = x[center_indices + 1]
            x = (x_src * x_dst)
            x = F.relu(self.lin1(x))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.lin2(x)

        return x
