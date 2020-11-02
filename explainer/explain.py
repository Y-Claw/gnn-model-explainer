""" explain.py

    Implementation of the explainer. 
"""

import math
import time
import os

import matplotlib
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

import networkx as nx
import numpy as np
import pandas as pd
import tensorboardX.utils

import torch
import torch.nn as nn
from torch.autograd import Variable

import sklearn.metrics as metrics
from sklearn.metrics import roc_auc_score, recall_score, precision_score, roc_auc_score, precision_recall_curve
from sklearn.cluster import DBSCAN

import pdb

import utils.io_utils as io_utils
import utils.train_utils as train_utils
import utils.graph_utils as graph_utils


use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
Tensor = FloatTensor


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
    ):
        self.model = model
        self.model.eval()
        self.graph = graph
        self.adj = adj
        self.feat = feat
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

    def extract_n_hops_neighborhood(self, node_idx):  # works for directed and undirected graph
        """Returns the n_hops neighborhood of a given ndoe."""
        neighbors_adj_row = self.neighborhoods[0][node_idx, :]          # in-edges
        neighbors_adj_column = self.neighborhoods[0][:, node_idx]       # out-edges
        n1 = np.nonzero(neighbors_adj_row)[1]
        n2 = np.nonzero(neighbors_adj_column)[0]
        neighbors = np.concatenate((n1, n2))
        neighbors = np.insert(neighbors, 0, node_idx)
        neighbors = np.unique(neighbors)

        node_idx_new = np.where(neighbors == node_idx)[0][0]
        sub_adj = self.adj.todense()[neighbors][:, neighbors]
        sub_feat = self.feat[0, neighbors]
        sub_node_label = self.node_labels[:, neighbors]

        return node_idx_new, sub_adj, sub_feat, sub_node_label, neighbors

    def explain_a_set_of_links(self, args):
        src_explain_res = []
        dst_explain_res = []
        denoise_res = []

        # all positive links
        labels = edges = pred = None
        if args.single_edge_label:  # there exists negative data
            labels = np.concatenate((self.test_labels[:, :int(self.test_labels.shape[1] / 2)],
                                    self.train_labels[:, :int(self.train_labels.shape[1] / 2)]), axis=1)
            edges = np.concatenate((self.test_idx[:int(self.test_labels.shape[1] / 2)],
                                    self.train_idx[:int(self.train_labels.shape[1] / 2)]), axis=0)
            pred = np.concatenate((self.pred_test[:, :int(self.test_labels.shape[1] / 2)],
                                    self.pred_train[:, :int(self.train_labels.shape[1] / 2)]), axis=1)
        elif args.multi_label or args.multi_class:
            labels = np.concatenate((self.test_labels, self.train_labels), axis=1)
            edges = np.concatenate((self.test_idx, self.train_idx), axis=0)
            pred = np.concatenate((self.pred_test, self.pred_train), axis=1)
        print("whole links num: ", edges.shape[0])

        for index in range(edges.shape[0]):
            print("For the " + str(index) + "-th link:")
            src_idx = edges[index][0]
            dst_idx = edges[index][1]
            link_label = labels[0][index]
            pred_label = pred[:, index]
            print("src_idx: ", src_idx, ", dst_idx: ", dst_idx)
            print("src node label: ", self.node_labels[0][src_idx])
            print("dst node label: ", self.node_labels[0][dst_idx])
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

        print("Have explained ", edges.shape[0], " links.")
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

        src_adj = torch.tensor(src_adj, dtype=torch.float)
        src_x = torch.tensor(src_sub_feat, requires_grad=True, dtype=torch.float)

        dst_adj = torch.tensor(dst_adj, dtype=torch.float)
        dst_x = torch.tensor(dst_sub_feat, requires_grad=True, dtype=torch.float)

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
        )
        if self.args.gpu:
            explainer = explainer.cuda()

        self.model.eval()

        explainer.train()
        begin_time = time.time()
        for epoch in range(self.args.num_epochs):
            explainer.zero_grad()
            explainer.optimizer.zero_grad()
            ypred = explainer(unconstrained=unconstrained)
            loss = explainer.loss(ypred, pred_label, epoch)
            loss.retain_grad()
            ypred.retain_grad()
            loss.backward()

            explainer.optimizer.step()
            if explainer.scheduler is not None:
                explainer.scheduler.step()

            src_mask_density, dst_mask_density = explainer.mask_density()
            if self.print_training:
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
        src_masked_adj = dst_masked_adj = src_masked_feat = dst_masked_feat = None
        if model == "exp":
            src_masked_adj = (
                explainer.src_masked_adj[0].cpu().detach().numpy()
                # explainer.src_mask.cpu().detach().numpy() * src_adj[0].cpu().detach().numpy()
            )
            dst_masked_adj = (
                explainer.dst_masked_adj[0].cpu().detach().numpy()
                # explainer.dst_mask.cpu().detach().numpy() * dst_adj[0].cpu().detach().numpy()
            )
            src_masked_feat = (
                explainer.src_masked_x[0].cpu().detach().numpy()
                # torch.sigmoid(explainer.src_feat_mask).cpu().detach().numpy() * src_x[0].cpu().detach().numpy()
            )
            dst_masked_feat = (
                explainer.dst_masked_x[0].cpu().detach().numpy()
                # torch.sigmoid(explainer.dst_feat_mask).cpu().detach().numpy() * dst_x[0].cpu().detach().numpy()
            )
        src_results = {
            "src_masked_feat": src_masked_feat,
            "src_masked_adj": src_masked_adj,
            "src_idx_new": src_idx_new,
            "src_sub_label": src_sub_label,
            "src_neighbors": src_neighbors,
        }
        dst_results = {
            "dst_masked_feat": dst_masked_feat,
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
    ):
        super(ExplainModule, self).__init__()
        self.src_adj = src_adj
        self.dst_adj = dst_adj
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

        init_strategy = "normal"
        src_num_nodes = src_adj.size()[1]
        dst_num_nodes = dst_adj.size()[1]
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
            "size": 0.001,
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
        mask = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes))
        # mask = torch.nn.init.constant_(mask, 0)
        if init_strategy == "normal":
            std = nn.init.calculate_gain("relu") * math.sqrt(
                2.0 / (num_nodes + num_nodes)
            )
            with torch.no_grad():
                mask.normal_(1.0, std)
                # mask.clamp_(0.0, 1.0)
        elif init_strategy == "const":
            nn.init.constant_(mask, const_val)

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
        src_masked_adj = src_adj * src_sym_mask
        dst_masked_adj = dst_adj * dst_sym_mask
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
        src_adj_sum = torch.sum(self.src_adj)
        dst_adj_sum = torch.sum(self.dst_adj)
        return src_mask_sum / src_adj_sum, dst_mask_sum / dst_adj_sum

    def forward(self, unconstrained=False, mask_features=True, marginalize=False):
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
                    self.src_masked_x = src_x * src_feat_mask
                    self.dst_masked_x = dst_x * dst_feat_mask
        self.src_masked_x.retain_grad()
        self.src_masked_adj.retain_grad()
        self.dst_masked_adj.retain_grad()

        ypred = self.model(self.src_masked_x, self.src_masked_adj, [self.src_idx_new, self.dst_idx_new], self.dst_masked_x, self.dst_masked_adj)
        return ypred

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
                if self.args.gpu:
                    pred_loss = torch.nn.functional.binary_cross_entropy(torch.sigmoid(pred), torch.sigmoid(pred_label.float()).cuda())
                else:
                    pred_loss = torch.nn.functional.binary_cross_entropy(torch.sigmoid(pred), torch.sigmoid(pred_label.float()))
            elif self.args.multi_label:
                if self.args.gpu:
                    pred_loss = torch.nn.functional.binary_cross_entropy(pred, pred_label.float().cuda())
                else:
                    pred_loss = torch.nn.functional.binary_cross_entropy(pred, pred_label.float())
        # size
        bias = torch.tensor(0.00001).cuda() if self.args.gpu else torch.tensor(0.00001)
        src_mask = self.src_mask
        dst_mask = self.dst_mask
        if self.mask_act == "sigmoid":
            src_mask = torch.sigmoid(self.src_mask)
            dst_mask = torch.sigmoid(self.dst_mask)
        elif self.mask_act == "ReLU":
            src_mask = nn.ReLU()(self.src_mask)
            dst_mask = nn.ReLU()(self.dst_mask)
        print(src_mask.device, self.src_adj.device)
        src_mask = src_mask*torch.squeeze(self.src_adj.cuda() if self.args.gpu else self.src_adj) + bias
        dst_mask = dst_mask*torch.squeeze(self.dst_adj.cuda() if self.args.gpu else self.dst_adj) + bias
        src_size_loss = self.coeffs["size"] * torch.sum(src_mask)
        dst_size_loss = self.coeffs["size"] * torch.sum(dst_mask)
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
        src_D = torch.diag(torch.sum(self.src_masked_adj[0], 0))
        src_m_adj = self.src_masked_adj if self.graph_mode else self.src_masked_adj[self.graph_idx]
        src_L = src_D - src_m_adj
        dst_D = torch.diag(torch.sum(self.dst_masked_adj[0], 0))
        dst_m_adj = self.dst_masked_adj if self.graph_mode else self.dst_masked_adj[self.graph_idx]
        dst_L = dst_D - dst_m_adj
        pred_label_t = torch.argmax(pred_label, 1)
        pred_label_t = torch.tensor(pred_label_t, dtype=torch.float)
        if self.args.gpu:
            pred_label_t = pred_label_t.cuda()
            src_L = src_L.cuda()
            dst_L = dst_L.cuda()
        if self.graph_mode:
            lap_loss = 0
        else:
            # wrong. cuase pred_label_t is the prediction result only for link(src, dst), not for all nodes in src_adj and dst_adj.
            # here the dimension of pred_label_t need to be equal to src_L and dst_L.
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

        loss = pred_loss + size_loss + lap_loss + mask_ent_loss + feat_size_loss

        #loss = pred_loss

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

