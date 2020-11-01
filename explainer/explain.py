""" explain.py

    Implementation of the explainer. 
"""

import math
import time
import os
import random
import matplotlib
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import tensorboardX.utils
import scipy.sparse as sp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import sklearn.metrics as metrics
from sklearn.metrics import roc_auc_score, recall_score, precision_score, roc_auc_score, precision_recall_curve
from sklearn.cluster import DBSCAN

import pdb

import utils.io_utils as io_utils
import utils.train_utils as train_utils
import utils.graph_utils as graph_utils

from torch_geometric.data import Data

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
Tensor = FloatTensor

import sys
path = os.path.join("..")
sys.path.append(path)
import PGNN_utils

class Explainer:
    def __init__(
        self,
        model,
        graph,
        adj,
        feat,
        node_labels,
        train_labels,
        test_labels,
        pred_train,
        pred_test,
        train_idx,
        test_idx,
        args,
        writer=None,
        print_training=True,
        graph_mode=False,
        graph_idx=False,
        label_dic=None,
    ):
        self.model = model
        self.model.eval()
        temp_graph = nx.DiGraph()
        temp_graph.add_nodes_from(sorted(list(graph.nodes)))
        for u, v in graph.edges:
            temp_graph.add_edges_from([[u,v,graph[u][v]]])
        self.graph = temp_graph
        self.adj = nx.to_scipy_sparse_matrix(self.graph)
        self.feat = feat
        for i in range(feat.shape[1]):
            self.graph.nodes[i]['label'] = feat[0][i][0]
        self.node_labels = node_labels
        self.train_labels = train_labels
        self.test_labels = test_labels
        self.pred_train = pred_train
        self.pred_test = pred_test
        self.train_idx = train_idx
        self.test_idx = test_idx
        # self.n_hops = args.num_gc_layers
        self.n_hops = args.n_hops
        self.graph_mode = graph_mode
        self.graph_idx = graph_idx
        self.neighborhoods = None if self.graph_mode else graph_utils.neighborhoods(adj=self.adj, n_hops=self.n_hops, use_cuda=use_cuda)
        self.args = args
        self.writer = writer
        self.print_training = print_training
        self.label_dic = label_dic

    def extract_n_hops_neighborhood(self, node_idx):  # works for directed and undirected graph
        """Returns the n_hops neighborhood of a given ndoe."""
        neighbors_adj_row = self.neighborhoods[0][node_idx, :]          # in-edges
        neighbors_adj_column = self.neighborhoods[0][:, node_idx]       # out-edges
        #n1 = np.nonzero(neighbors_adj_row)[0]
        n1 = np.nonzero(neighbors_adj_row)[1]
        n2 = np.nonzero(neighbors_adj_column)[0]
        neighbors = np.concatenate((n1, n2))
        neighbors = np.insert(neighbors, 0, node_idx)
        neighbors = np.unique(neighbors)

        node_idx_new = np.where(neighbors == node_idx)[0][0]
        #sub_adj = self.adj[0][neighbors][:, neighbors]
        sub_adj = self.adj.tocsc()[neighbors][:, neighbors]
        sub_feat = self.feat[0, neighbors]
        sub_node_label = self.node_labels[:, neighbors]

        return node_idx_new, sub_adj, sub_feat, sub_node_label, neighbors

    def explain_a_set_of_links(self, args):
        #dir = os.path.join("data","cit_patents_sp")
        #name = "cit_patents_0"
        dir = "data"
        name = "USAir"
        #G, _, _, num_edge_labels = io_utils.read_graphfile(datadir=dir, dataname=name,args=args)
        G = self.graph
        num_edge_labels=1
        """G.remove_edges_from(nx.selfloop_edges(G))
        # relabel graphs
        keys = list(G.nodes)
        vals = range(G.number_of_nodes())
        mapping = dict(zip(keys, vals))
        nx.relabel_nodes(G, mapping, copy=False)"""

        src_explain_res = []
        dst_explain_res = []

        #src_denoise_res = []
        #dst_denoise_res = []

        denoise_res = []

        # all positive links
        labels = edges = pred = None
        if args.single_edge_label:  # there exists negative data
            labels = np.concatenate((self.test_labels[:, :int(self.test_labels.shape[1] / 4)],
                                    self.train_labels[:, :int(self.train_labels.shape[1] / 2)]), axis=1)
            edges = np.concatenate((self.test_idx[:int(self.test_labels.shape[1] / 4)],
                                    self.train_idx[:int(self.train_labels.shape[1] / 2)]), axis=0)
            pred = np.concatenate((self.pred_test[:, :int(self.test_labels.shape[1] / 4)],
                                    self.pred_train[:, :int(self.train_labels.shape[1] / 2)]), axis=1)
        elif args.multi_label or args.multi_class:
            labels = np.concatenate((self.test_labels, self.train_labels), axis=1)
            edges = np.concatenate((self.test_idx, self.train_idx), axis=0)
            pred = np.concatenate((self.pred_test, self.pred_train), axis=1)

        explain_list = []

        explain_list.append((220,251))

        label_dic = {}

        for i in range(num_edge_labels):
            label_dic[i] = []

        for u, v in edges:
            label_dic[G.get_edge_data(u, v)['label']].append((u, v))

        for l in label_dic.values():
            explain_list += random.sample(l, args.sample_num)

        edges_list = edges.tolist()
        nodes_list = list(G.nodes)

        for u,v in explain_list:
        #for index in range(edges.shape[0]):
            index = edges_list.index([u,v])
            src_idx = edges[index][0]
            #src_idx = nodes_list.index(src_idx)
            dst_idx = edges[index][1]
            #dst_idx = nodes_list.index(dst_idx)
            link_label = labels[0][index]
            pred_label = pred[:, index]
            #if (src_idx != 44 or dst_idx != 57) and (src_idx != 57 or dst_idx != 44):
            #    continue
            #if math.fabs(pred_label[0][0]) < 0.6:
            #    continue
            print("src_idx: ", src_idx, ", dst_idx: ", dst_idx)
            print("src node label: ", self.node_labels[0][src_idx])
            print("dst node label: ", self.node_labels[0][dst_idx])
            """src_results, dst_results = self.explain_link(
                    src_idx, dst_idx, link_label, pred_label, args
            )
            src_explain_res.append(src_results)
            dst_explain_res.append(dst_results)

            src_masked_feat = src_results["src_sub_feat"]
            src_masked_adj = src_results["src_masked_adj"]
            src_idx_new = src_results["src_idx_new"]
            src_neighbors = src_results["src_neighbors"]

            dst_masked_feat = dst_results["dst_sub_feat"]
            dst_masked_adj = dst_results["dst_masked_adj"]
            dst_idx_new = dst_results["dst_idx_new"]
            dst_neighbors = dst_results["dst_neighbors"]

            src_denoise_result = io_utils.denoise_adj_feat(
                self.graph, src_masked_adj, src_idx_new, src_masked_feat, src_neighbors,
                edge_threshold=args.edge_threshold, feat_threshold=args.feat_threshold, edge_num_threshold=args.edge_num_threshold_src_or_dst, args=args
            )
            if src_denoise_result is None:
                continue
            dst_denoise_result = io_utils.denoise_adj_feat(
                self.graph, dst_masked_adj, dst_idx_new, dst_masked_feat, dst_neighbors,
                edge_threshold=args.edge_threshold, feat_threshold=args.feat_threshold, edge_num_threshold=args.edge_num_threshold_src_or_dst, args=args
            )
            if dst_denoise_result is None:
                continue
            src_denoise_res.append(src_denoise_result)
            dst_denoise_res.append(dst_denoise_result)

            io_utils.combine_src_dst_explanations(self.graph, index, src_idx, dst_idx, link_label, src_denoise_result, dst_denoise_result, args)
            #break"""
            src_explanation_results, dst_explanation_results = self.explain_link(
                src_idx, dst_idx, link_label, pred_label, args
            )
            src_explain_res.append(src_explanation_results)
            dst_explain_res.append(dst_explanation_results)

            src_dst_explanation = io_utils.combine_src_dst_explanations(
                src_explanation_results, dst_explanation_results
            )

            denoise_result = io_utils.denoise_adj_feat(
                self.graph, index, src_dst_explanation, link_label,
                edge_threshold=args.edge_threshold, feat_threshold=args.feat_threshold,
                edge_num_threshold=args.max_edges_num, args=args
            )

            denoise_res.append(denoise_result)

        #return src_explain_res, dst_explain_res, src_denoise_res, dst_denoise_res
        return src_explain_res, dst_explain_res, denoise_res


    def explain_link(self, src_idx, dst_idx, link_label, pred_label, args, unconstrained=False, model="exp"):
        """Explain link prediction for a single node pair
        """
        src_idx_new, src_adj, src_sub_feat, src_sub_label, src_neighbors = self.extract_n_hops_neighborhood(
            src_idx
        )
        dst_idx_new, dst_adj, dst_sub_feat, dst_sub_label, dst_neighbors = self.extract_n_hops_neighborhood(
            dst_idx
        )
        src_adj = np.expand_dims(src_adj, axis=0)
        dst_adj = np.expand_dims(dst_adj, axis=0)
        src_sub_feat = np.expand_dims(src_sub_feat, axis=0)
        dst_sub_feat = np.expand_dims(dst_sub_feat, axis=0)

        #src_adj = torch.tensor(src_adj, dtype=torch.float)
        src_adj = io_utils.sparse_mx_to_torch_sparse_tensor(src_adj[0])
        src_x = torch.tensor(src_sub_feat, requires_grad=True, dtype=torch.float)
        #src_x = io_utils.sparse_mx_to_torch_sparse_tensor(src_sub_feat[0])


        #dst_adj = torch.tensor(dst_adj, dtype=torch.float)
        dst_adj = io_utils.sparse_mx_to_torch_sparse_tensor(dst_adj[0])
        dst_x = torch.tensor(dst_sub_feat, requires_grad=True, dtype=torch.float)
        #dst_x = io_utils.sparse_mx_to_torch_sparse_tensor(dst_sub_feat[0])
        print("link label:", link_label)

        pred_label = torch.tensor(pred_label)
        print("link predicted label: ", pred_label)

        explainer = ExplainModule(
            src_adj=src_adj,
            dst_adj=dst_adj,
            src_x=src_x,
            dst_x=dst_x,
            src_idx_new=src_idx_new,
            dst_idx_new=dst_idx_new,
            model=self.model,
            link_label=link_label,
            args=self.args,
            writer=self.writer,
            graph_idx=self.graph_idx,
            graph_mode=self.graph_mode,
            mode=self.args.model,
            node_num=self.adj.shape[1]
        )
        if self.args.gpu:
            explainer = explainer.cuda()

        self.model.eval()

        explainer.train()
        begin_time = time.time()
        clean = True
        device = torch.device('cuda:' + str(self.args.cuda) if self.args.gpu else 'cpu')
        for epoch in range(self.args.num_epochs):
            start = time.time()
            explainer.zero_grad()
            explainer.optimizer.zero_grad()
            explain_s = time.time()
            ypred, src_adj_atts, dst_adj_atts = explainer(unconstrained=unconstrained)
            explain_e = time.time()
            loss_s = time.time()
            loss = explainer.loss(ypred, pred_label.to(device), epoch)
            loss_e = time.time()
            ypred.retain_grad()
            #explainer.nodes_first.retain_grad()
            #if self.args.model == "GCN":
            #    for data in explainer.data:
            #        for var in data.edge_attr:
            #            var.retain_grad()
            back_s = time.time()
            loss.backward()
            back_e = time.time()

            explainer.optimizer.step()
            if explainer.scheduler is not None:
                explainer.scheduler.step()
            #if clean and epoch % 20 == 0 and epoch != 0:
                #explainer.clean_mask()
                #explainer.train()
            end = time.time()
            #print("all: ", end - start, "backward: ", back_e - back_s, "explain: ", explain_e - explain_s, "loss: ", loss_e - loss_s)
            src_mask_density, dst_mask_density = explainer.mask_density()
            if self.print_training and epoch % 10 == 0:
                print(
                    "epoch: ",
                    epoch,
                    "; loss: ",
                    loss.item(),
                    "; src mask density: ",
                    src_mask_density.item(),
                    "; dst mask density: ",
                    dst_mask_density.item(),
                    "; pred: ",
                    ypred,
                )

            # if self.writer is not None:
            #         self.writer.add_scalar("mask/src_density", src_mask_density, epoch)
            #         self.writer.add_scalar("mask/dst_density", dst_mask_density, epoch)
            #         self.writer.add_scalar(
            #             "optimization/lr",
            #             explainer.optimizer.param_groups[0]["lr"],
            #             epoch,
            #         )
            #         # if epoch % 25 == 0:
            #         #     explainer.log_mask(epoch)
            #         #     explainer.log_masked_adj(
            #         #         node_idx_new, epoch, label=single_subgraph_label
            #         #     )
            #         #     explainer.log_adj_grad(
            #         #         node_idx_new, pred_label, epoch, label=single_subgraph_label
            #         #     )
            #
            #         # if epoch == 0:
            #         #     if self.model.att:
            #         #         # explain node
            #         #         print("adj att size: ", adj_atts.size())
            #         #         adj_att = torch.sum(adj_atts[0], dim=2)
            #         #         # adj_att = adj_att[neighbors][:, neighbors]
            #         #         node_adj_att = adj_att * adj.float().cuda()
            #         #         io_utils.log_matrix(
            #         #             self.writer, node_adj_att[0], "att/matrix", epoch
            #         #         )
            #         #         node_adj_att = node_adj_att[0].cpu().detach().numpy()
            #         #         G = io_utils.denoise_graph(
            #         #             node_adj_att,
            #         #             node_idx_new,
            #         #             threshold=3.8,  # threshold_num=20,
            #         #             max_component=True,
            #         #         )
            #         #         io_utils.log_graph(
            #         #             self.writer,
            #         #             G,
            #         #             name="att/graph",
            #         #             identify_self=not self.graph_mode,
            #         #             nodecolor="label",
            #         #             edge_vmax=None,
            #         #             args=self.args,
            #         #         )
            if model != "exp":
                break

        print("finished training in ", time.time() - begin_time)
        if model == "exp":
            """src_masked_adj = (
                explainer.src_masked_adj[0].cpu().detach().numpy() * src_adj[0].cpu().detach().numpy()
            )
            dst_masked_adj = (
                explainer.dst_masked_adj[0].cpu().detach().numpy() * dst_adj[0].cpu().detach().numpy()
            )"""
            src_masked_adj = (
                    explainer.src_masked_adj.cpu().detach().numpy() * src_adj.cpu().detach().to_dense().numpy()
            )
            dst_masked_adj = (
                    explainer.dst_masked_adj.cpu().detach().numpy() * dst_adj.cpu().detach().to_dense().numpy()
            )
            src_sub_feat = (
                torch.sigmoid(explainer.src_feat_mask).cpu().detach().numpy() * src_x[0].cpu().detach().numpy()
                # explainer.src_feat_mask.cpu().detach().numpy() * src_x[0].cpu().detach().numpy()
            )
            dst_sub_feat = (
                torch.sigmoid(explainer.dst_feat_mask).cpu().detach().numpy() * dst_x[0].cpu().detach().numpy()
                # explainer.dst_feat_mask.cpu().detach().numpy() * dst_x[0].cpu().detach().numpy()
            )
        # else:
        #     adj_atts = nn.functional.sigmoid(adj_atts).squeeze()
        #     masked_adj = adj_atts.cpu().detach().numpy() * sub_adj.squeeze()
        #
        fname = 'masked_adj_' + io_utils.gen_explainer_prefix(self.args) + (
                'src_idx_'+str(src_idx)+'dst_idx_'+str(dst_idx)+'.npy')
        # with open(os.path.join(self.args.logdir, fname), 'wb') as outfile:
        #     np.save(outfile, np.asarray(src_masked_adj.copy()))
        #     np.save(outfile, np.asarray(dst_masked_adj.copy()))
        #     np.save(outfile, np.array(src_sub_feat.copy()))
        #     np.save(outfile, np.array(dst_sub_feat.copy()))
        #     print("Saved adjacency matrix to ", fname)

        src_results = {
            "src_masked_feat": src_sub_feat,
            "src_masked_adj": src_masked_adj,
            "src_idx_new": src_idx_new,
            "src_sub_label": src_sub_label,
            "src_neighbors": src_neighbors,
        }
        dst_results = {
            "dst_masked_feat": dst_sub_feat,
            "dst_masked_adj": dst_masked_adj,
            "dst_idx_new": dst_idx_new,
            "dst_sub_label": dst_sub_label,
            "dst_neighbors": dst_neighbors,
        }
        return src_results, dst_results


class ExplainModule(nn.Module):
    def __init__(
        self,
        src_adj,
        dst_adj,
        src_x,
        dst_x,
        src_idx_new,
        dst_idx_new,
        model,
        link_label,
        args,
        graph_idx=0,
        writer=None,
        use_sigmoid=True,
        graph_mode=False,
        mode="default",
        threshold=0.2,
        node_num=None,
    ):
        super(ExplainModule, self).__init__()
        self.src_adj = src_adj
        self.dst_adj = dst_adj
        #self.src_adj = io_utils.sparse_mx_to_torch_sparse_tensor(src_adj[0])
        #self.dst_adj = io_utils.sparse_mx_to_torch_sparse_tensor(dst_adj[0])
        self.src_x = src_x
        self.dst_x = dst_x
        self.src_idx_new=src_idx_new
        self.dst_idx_new=dst_idx_new
        self.model = model
        self.link_label = link_label
        self.graph_idx = graph_idx
        self.args = args
        self.writer = writer
        self.mask_act = args.mask_act
        self.use_sigmoid = use_sigmoid
        self.graph_mode = graph_mode
        self.mode = mode
        self.threshold = threshold
        self.node_num = node_num

        init_strategy = "normal"
        #src_num_nodes = src_adj.size()[1]
        #dst_num_nodes = dst_adj.size()[1]
        src_num_nodes = src_adj[0].shape[0]
        dst_num_nodes = dst_adj[0].shape[0]
        self.src_mask, self.src_mask_bias = self.construct_edge_mask(
            src_num_nodes, init_strategy=init_strategy
        )
        self.dst_mask, self.dst_mask_bias = self.construct_edge_mask(
            dst_num_nodes, init_strategy=init_strategy
        )

        self.src_feat_mask = self.construct_feat_mask(src_x.size(-1), init_strategy="constant")
        self.dst_feat_mask = self.construct_feat_mask(dst_x.size(-1), init_strategy="constant")

        params = [self.src_mask, self.src_feat_mask, self.dst_mask, self.dst_feat_mask]
        if self.src_mask_bias is not None:
            params.append(self.src_mask_bias)
        if self.dst_mask_bias is not None:
            params.append(self.dst_mask_bias)
        # For masking diagonal entries
        self.src_diag_mask = torch.ones(src_num_nodes, src_num_nodes) - torch.eye(src_num_nodes)
        self.dst_diag_mask = torch.ones(dst_num_nodes, dst_num_nodes) - torch.eye(dst_num_nodes)
        if args.gpu:
            self.src_diag_mask = self.src_diag_mask.cuda()
            self.dst_diag_mask = self.dst_diag_mask.cuda()

        self.scheduler, self.optimizer = train_utils.build_optimizer(args, params)

        self.coeffs = {
            "size": 0.005 if mode == "default" else 0.005,
            "feat_size": 1.0,
            "ent": 1.0,
            "feat_ent": 0.1,
            "grad": 0,
            "lap": 1.0,
        }

    def construct_feat_mask(self, feat_dim, init_strategy="normal"):
        mask = nn.Parameter(torch.FloatTensor(feat_dim))
        if init_strategy == "normal":
            std = 0.1
            with torch.no_grad():
                mask.normal_(1.0, std)
        elif init_strategy == "constant":
            with torch.no_grad():
                nn.init.constant_(mask, 0.0)
                # mask[0] = 2
        return mask

    def construct_edge_mask(self, num_nodes, init_strategy="normal", const_val=1.0):
        #sparse_mask = sp.rand(num_nodes,num_nodes,density=0.5,format='coo')
        mask = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes))
        if init_strategy == "normal":
            std = nn.init.calculate_gain("relu") * math.sqrt(
                2.0 / (num_nodes + num_nodes)
            )
            with torch.no_grad():
                mask.normal_(1.0, std)
                # mask.clamp_(0.0, 1.0)
        elif init_strategy == "const":
            nn.init.constant_(mask, const_val)
        #nn.init.constant_(mask, 1)
        if self.args.mask_bias:
            mask_bias = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes))
            nn.init.constant_(mask_bias, 0.0)
        else:
            mask_bias = None

        return mask, mask_bias

    def _masked_adj(self):
        src_sym_mask = self.src_mask
        dst_sym_mask = self.dst_mask
        if self.mask_act == "sigmoid":
            src_sym_mask = torch.sigmoid(self.src_mask)
            dst_sym_mask = torch.sigmoid(self.dst_mask)
        elif self.mask_act == "ReLU":
            src_sym_mask = nn.ReLU()(self.src_mask)
            dst_sym_mask = nn.ReLU()(self.dst_mask)
        src_sym_mask = (src_sym_mask + src_sym_mask.t()) / 2
        dst_sym_mask = (dst_sym_mask + dst_sym_mask.t()) / 2
        src_adj = self.src_adj.cuda() if self.args.gpu else self.src_adj
        dst_adj = self.dst_adj.cuda() if self.args.gpu else self.dst_adj
        src_masked_adj = src_adj.to_dense() * src_sym_mask
        dst_masked_adj = dst_adj.to_dense() * dst_sym_mask
        if self.args.mask_bias:
            src_bias = (self.src_mask_bias + self.src_mask_bias.t()) / 2
            dst_bias = (self.dst_mask_bias + self.dst_mask_bias.t()) / 2
            src_bias = nn.ReLU6()(src_bias * 6) / 6
            dst_bias = nn.ReLU6()(dst_bias * 6) / 6
            src_masked_adj += (src_bias + src_bias.t()) / 2
            dst_masked_adj += (dst_bias + dst_bias.t()) / 2
        return src_masked_adj * self.src_diag_mask, dst_masked_adj * self.dst_diag_mask

    def mask_density(self):
        src_masked_adj, dst_masked_adj = self._masked_adj()
        src_mask_sum = torch.sum(src_masked_adj).cpu()
        dst_mask_sum = torch.sum(dst_masked_adj).cpu()
        #src_adj_sum = torch.sum(self.src_adj)
        #dst_adj_sum = torch.sum(self.dst_adj)
        src_adj_sum = self.src_adj.coalesce().indices().shape[1]
        dst_adj_sum = self.dst_adj.coalesce().indices().shape[1]
        return src_mask_sum / src_adj_sum, dst_mask_sum / dst_adj_sum

    def out(self, x, adj, device, mode="GCN", num_node = None):
        edges = []
        """for i in range(adj.shape[1]):
            for j in range(adj.shape[2]):
                if adj[0][i][j] > 0:
                    edges.append([i, j])"""
        if mode == "PGNN":
            for i in range(adj.shape[0]):
                for j in range(adj.shape[1]):
                    if adj[i,j] > 0:
                        edges.append([i, j])
            edges = np.array(edges)
        else:
            edges = np.nonzero(adj.squeeze(0).detach().numpy())
            edges = np.swapaxes(edges, -2, -1)
        edge_index = np.concatenate((edges, edges[:, ::-1]), axis=0)
        edge_index = torch.from_numpy(edge_index).long().permute(1, 0)
        edge_weight = [adj[i,j].unsqueeze(0) for [i, j] in edge_index.permute(1, 0)]
        try:
            edge_weight = torch.cat(edge_weight)
        except:
            edge_weight = torch.cat(edge_weight)
        self.edge_weight = edge_weight
        #edge_weight.requires_grad = True
        data = Data(x=x.squeeze(0), edge_index=edge_index,
                    edge_attr=edge_weight,
                    )
        self.data = data
        if self.mode == "PGNN":
            data.dists = PGNN_utils.precompute_dist_data(edge_index, edge_weight, adj.shape[1])
            data.num_nodes = adj.shape[1]
            PGNN_utils.preselect_anchor(data, device=device, set_num=num_node)#change args to set anchor num
        return self.model(data, explain=True) if self.args.save == "pred" else self.model.get_embedding(data, explain=True)

    def pred(self, out, src, dst):
        nodes_first = torch.index_select(out, 0, torch.from_numpy(edge_mask_train[0, :]).long().to(device))
        nodes_second = torch.index_select(out, 0, torch.from_numpy(edge_mask_train[1, :]).long().to(device))
        pred = torch.sum(nodes_first * nodes_second, dim=-1)

    def clean_mask(self):
        value = -10
        src_mask = self.src_mask
        dst_mask = self.dst_mask
        if self.mask_act == "sigmoid":
            src_mask = torch.sigmoid(self.src_mask)
            dst_mask = torch.sigmoid(self.dst_mask)
        elif self.mask_act == "ReLU":
            src_mask = nn.ReLU()(self.src_mask)
            dst_mask = nn.ReLU()(self.dst_mask)
        for i in range(src_mask.shape[0]):
            for j in range(src_mask.shape[1]):
                if src_mask[i][j] < self.threshold:
                    self.src_mask[i][j] = value
        for i in range(dst_mask.shape[0]):
            for j in range(dst_mask.shape[1]):
                if dst_mask[i][j] < self.threshold:
                    self.dst_mask[i][j] = value


    def forward(self, unconstrained=False, mask_features=True, marginalize=False, clean=False):
        start = time.time()
        src_adj_att = None
        dst_adj_att = None
        ypred = None

        src_x = self.src_x.cuda() if self.args.gpu else self.src_x
        dst_x = self.dst_x.cuda() if self.args.gpu else self.dst_x

        if unconstrained:
            sym_mask = torch.sigmoid(self.mask) if self.use_sigmoid else self.mask
            self.masked_adj = (
                torch.unsqueeze((sym_mask + sym_mask.t()) / 2, 0) * self.diag_mask
            )
        else:
            self.src_masked_adj, self.dst_masked_adj = self._masked_adj()
            if mask_features:
                src_feat_mask = (
                    torch.sigmoid(self.src_feat_mask)
                    if self.use_sigmoid
                    else self.src_feat_mask
                )
                dst_feat_mask = (
                    torch.sigmoid(self.dst_feat_mask)
                    if self.use_sigmoid
                    else self.dst_feat_mask
                )
                if marginalize:
                    src_std_tensor = torch.ones_like(src_x, dtype=torch.float) / 2
                    src_mean_tensor = torch.zeros_like(src_x, dtype=torch.float) - src_x
                    src_z = torch.normal(mean=src_mean_tensor, std=src_std_tensor)
                    src_x = src_x + src_z * (1 - src_feat_mask)
                    dst_std_tensor = torch.ones_like(dst_x, dtype=torch.float) / 2
                    dst_mean_tensor = torch.zeros_like(dst_x, dtype=torch.float) - dst_x
                    dst_z = torch.normal(mean=dst_mean_tensor, std=dst_std_tensor)
                    dst_x = dst_x + dst_z * (1 - dst_feat_mask)
                else:
                    src_x = src_x * src_feat_mask
                    dst_x = dst_x * dst_feat_mask
        if self.mode == "default":
            ypred, src_adj_att, dst_adj_att = self.model(src_x, self.src_masked_adj, [self.src_idx_new, self.dst_idx_new], dst_x, self.dst_masked_adj)
        else:
            device = torch.device('cuda:' + str(self.args.cuda) if self.args.gpu else 'cpu')
            num_node = None
            if self.mode == "PGNN":
                num_node = self.node_num
            src_out = self.out(src_x, self.src_masked_adj, device, mode=self.mode, num_node=num_node)
            dst_out = self.out(dst_x, self.dst_masked_adj, device, mode=self.mode, num_node=num_node)
            self.src_out = src_out
            self.src_masked_adj.retain_grad()
            nodes_first = torch.index_select(src_out, 0, torch.from_numpy(np.array([self.src_idx_new])).long().to(device))
            nodes_second = torch.index_select(dst_out, 0, torch.from_numpy(np.array([self.dst_idx_new])).long().to(device))
            self.nodes_first = nodes_first
            ypred = torch.sum(nodes_first * nodes_second, dim=-1) if self.args.save == "pred" else torch.cat((nodes_first, nodes_second), dim=-1)

        end = time.time()
        #print("explain_forward: ", end - start)
        return ypred, src_adj_att, dst_adj_att

    def loss(self, pred, pred_label, epoch):
        """
        Args:
            pred: prediction made by current model
            pred_label: the label predicted by the original model.
        """
        mi_obj = False
        pred_loss = 0
        if mi_obj:
            pred_loss = -torch.sum(pred * torch.log(pred))
        else:
            # pred_label_node = pred_label if self.graph_mode else pred_label[node_idx]
            # gt_label_node = self.label if self.graph_mode else self.label[0][node_idx]
            # logit = pred[gt_label_node]
            # pred_loss = -torch.log(logit)
            if self.args.single_edge_label or self.args.multi_class:
                if self.mode == "default":
                    pred_loss = torch.nn.functional.binary_cross_entropy(torch.sigmoid(pred), torch.sigmoid(pred_label.float()))
                elif self.mode == "GCN" or self.mode == "PGNN":
                    loss_fuc = torch.nn.BCELoss()
                    pred_loss = F.l1_loss(pred, pred_label)
                elif self.args.multi_label:
                    pred_loss = torch.nn.functional.binary_cross_entropy(pred, pred_label.float())
        # size
        src_mask = self.src_mask
        dst_mask = self.dst_mask
        if self.mask_act == "sigmoid":
            src_mask = torch.sigmoid(self.src_mask)
            dst_mask = torch.sigmoid(self.dst_mask)
        elif self.mask_act == "ReLU":
            src_mask = nn.ReLU()(self.src_mask)
            dst_mask = nn.ReLU()(self.dst_mask)
        zero = torch.zeros(src_mask.shape)
        src_size_loss = self.coeffs["size"] * torch.sum(src_mask.where(src_mask > self.threshold, zero))
        zero = torch.zeros(dst_mask.shape)
        dst_size_loss = self.coeffs["size"] * torch.sum(dst_mask.where(dst_mask > self.threshold, zero))
        size_loss = src_size_loss + dst_size_loss

        # pre_mask_sum = torch.sum(self.feat_mask)
        src_feat_mask = (
            torch.sigmoid(self.src_feat_mask) if self.use_sigmoid else self.src_feat_mask
        )
        dst_feat_mask = (
            torch.sigmoid(self.dst_feat_mask) if self.use_sigmoid else self.dst_feat_mask
        )
        src_feat_size_loss = self.coeffs["feat_size"] * torch.mean(src_feat_mask)
        dst_feat_size_loss = self.coeffs["feat_size"] * torch.mean(dst_feat_mask)
        feat_size_loss = src_feat_size_loss + dst_feat_size_loss

        # entropy
        src_mask_ent = -src_mask * torch.log(src_mask) - (1 - src_mask) * torch.log(1 - src_mask)
        dst_mask_ent = -dst_mask * torch.log(dst_mask) - (1 - dst_mask) * torch.log(1 - dst_mask)
        src_mask_ent_loss = self.coeffs["ent"] * torch.mean(src_mask_ent)
        dst_mask_ent_loss = self.coeffs["ent"] * torch.mean(dst_mask_ent)
        mask_ent_loss = src_mask_ent_loss + dst_mask_ent_loss

        src_feat_mask_ent = - src_feat_mask             \
                            * torch.log(src_feat_mask)  \
                            - (1 - src_feat_mask)       \
                            * torch.log(1 - src_feat_mask)
        dst_feat_mask_ent = - dst_feat_mask \
                            * torch.log(dst_feat_mask) \
                            - (1 - dst_feat_mask) \
                            * torch.log(1 - dst_feat_mask)

        src_feat_mask_ent_loss = self.coeffs["feat_ent"] * torch.mean(src_feat_mask_ent)
        dst_feat_mask_ent_loss = self.coeffs["feat_ent"] * torch.mean(dst_feat_mask_ent)
        feat_mask_ent_loss = src_feat_mask_ent_loss + dst_feat_mask_ent_loss

        # laplacian
        #src_D = torch.diag(torch.sum(self.src_masked_adj[0], 0))
        src_D = torch.diag(torch.sum(self.src_masked_adj, 0))
        src_m_adj = self.src_masked_adj if self.graph_mode else self.src_masked_adj[self.graph_idx]
        src_L = src_D - src_m_adj
        #dst_D = torch.diag(torch.sum(self.dst_masked_adj[0], 0))
        dst_D = torch.diag(torch.sum(self.dst_masked_adj, 0))
        dst_m_adj = self.dst_masked_adj if self.graph_mode else self.dst_masked_adj[self.graph_idx]
        dst_L = dst_D - dst_m_adj
        pred_label_t = torch.tensor(pred_label, dtype=torch.float)
        if self.args.gpu:
            pred_label_t = pred_label_t.cuda()
            src_L = src_L.cuda()
            dst_L = dst_L.cuda()
        if self.graph_mode:
            lap_loss = 0
        else:
            # src_lap_loss = (self.coeffs["lap"]
            #     * (pred_label_t @ src_L @ pred_label_t)
            #     / self.src_adj.numel()
            # )
            # dst_lap_loss = (self.coeffs["lap"]
            #     * (pred_label_t @ dst_L @ pred_label_t)
            #     / self.dst_adj.numel()
            # )
            # lap_loss = src_lap_loss + dst_lap_loss
            lap_loss = 0
        # grad
        # adj
        # adj_grad, x_grad = self.adj_feat_grad(node_idx, pred_label_node)
        # adj_grad = adj_grad[self.graph_idx]
        # x_grad = x_grad[self.graph_idx]
        # if self.args.gpu:
        #    adj_grad = adj_grad.cuda()
        # grad_loss = self.coeffs['grad'] * -torch.mean(torch.abs(adj_grad) * mask)

        # feat
        # x_grad_sum = torch.sum(x_grad, 1)
        # grad_feat_loss = self.coeffs['featgrad'] * -torch.mean(x_grad_sum * mask)

        #loss = pred_loss + size_loss + lap_loss + mask_ent_loss + feat_size_loss
        #loss = pred_loss + lap_loss + mask_ent_loss + feat_size_loss
        loss = pred_loss + lap_loss + feat_size_loss
        if self.writer is not None:
            self.writer.add_scalar("optimization/size_loss", size_loss, epoch)
            self.writer.add_scalar("optimization/feat_size_loss", feat_size_loss, epoch)
            self.writer.add_scalar("optimization/mask_ent_loss", mask_ent_loss, epoch)
            self.writer.add_scalar(
                "optimization/feat_mask_ent_loss", mask_ent_loss, epoch
            )
            # self.writer.add_scalar('optimization/grad_loss', grad_loss, epoch)
            self.writer.add_scalar("optimization/pred_loss", pred_loss, epoch)
            self.writer.add_scalar("optimization/lap_loss", lap_loss, epoch)
            self.writer.add_scalar("optimization/overall_loss", loss, epoch)
        return loss

