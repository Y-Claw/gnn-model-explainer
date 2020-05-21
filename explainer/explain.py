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
import seaborn as sns
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
        self.n_hops = 1
        self.graph_mode = graph_mode
        self.graph_idx = graph_idx
        self.neighborhoods = None if self.graph_mode else graph_utils.neighborhoods(adj=self.adj, n_hops=self.n_hops, use_cuda=use_cuda)
        self.args = args
        self.writer = writer
        self.print_training = print_training

    def extract_neighborhood_in_directed_graph(self, node_idx, graph_idx=0):
        """Returns the neighborhood of a given ndoe."""
        neighbors_adj_row = self.neighborhoods[graph_idx][node_idx, :]          # in-edges
        neighbors_adj_column = self.neighborhoods[graph_idx][:, node_idx]       # out-edges
        n1 = np.nonzero(neighbors_adj_row)[0]
        n2 = np.nonzero(neighbors_adj_column)[0]
        neighbors = np.concatenate((n1, n2))
        neighbors = np.insert(neighbors, 0, node_idx)
        neighbors = np.unique(neighbors)

        node_idx_new = np.where(neighbors == node_idx)[0][0]
        sub_adj = self.adj[graph_idx][neighbors][:, neighbors]
        sub_feat = self.feat[graph_idx, neighbors]
        sub_node_label = self.node_labels[:, neighbors]

        return node_idx_new, sub_adj, sub_feat, sub_node_label, neighbors

    def explain_a_set_of_links(self, args):
        src_explain_res = []
        dst_explain_res = []

        src_denoise_res = []
        dst_denoise_res = []

        labels = self.test_labels[:, :10]
        edges = self.test_idx[:10]
        pred = self.pred_test[:, :10]
        # labels = self.train_labels[:, :10]
        # pred = self.pred_train[:, :10]
        # edges = self.train_idx[:10]

        for index in range(edges.shape[0]):
            src_idx = edges[index][0]
            dst_idx = edges[index][1]
            link_label = labels[0][index]
            pred_label = pred[:, index]
            print("src_idx: ", src_idx, ", dst_idx: ", dst_idx)
            print("src node label: ", self.node_labels[0][src_idx])
            print("dst node label: ", self.node_labels[0][dst_idx])
            src_results, dst_results = self.explain_link(
                    src_idx, dst_idx, link_label, pred_label, args
            )
            src_explain_res.append(src_results)
            dst_explain_res.append(dst_results)

            src_sub_feat = src_results["src_sub_feat"]
            src_masked_adj = src_results["src_masked_adj"]
            src_idx_new = src_results["src_idx_new"]
            src_neighbors = src_results["src_neighbors"]

            dst_sub_feat = dst_results["dst_sub_feat"]
            dst_masked_adj = dst_results["dst_masked_adj"]
            dst_idx_new = dst_results["dst_idx_new"]
            dst_neighbors = dst_results["dst_neighbors"]

            src_denoise_result = io_utils.denoise_adj_feat(
                self.graph, src_masked_adj, src_idx_new, src_sub_feat, src_neighbors,
                edge_threshold=0.001, feat_threshold=0.001
            )
            dst_denoise_result = io_utils.denoise_adj_feat(
                self.graph, dst_masked_adj, dst_idx_new, dst_sub_feat, dst_neighbors,
                edge_threshold=0.0001, feat_threshold=0.001
            )
            src_denoise_res.append(src_denoise_result)
            dst_denoise_res.append(dst_denoise_result)

        return src_explain_res, dst_explain_res, src_denoise_res, dst_denoise_res

    def explain_link(self, src_idx, dst_idx, link_label, pred_label, args, unconstrained=False, model="exp"):
        """Explain link prediction for a single node pair
        """
        # index = 1
        # src_idx = self.train_idx[index][0]
        # dst_idx = self.train_idx[index][1]
        # link_label = self.train_labels[0][index]
        # print("src node label: ", self.node_labels[0][src_idx])
        # print("dst node label: ", self.node_labels[0][dst_idx])
        src_idx_new, src_adj, src_sub_feat, src_sub_label, src_neighbors = self.extract_neighborhood_in_directed_graph(
            src_idx
        )
        dst_idx_new, dst_adj, dst_sub_feat, dst_sub_label, dst_neighbors = self.extract_neighborhood_in_directed_graph(
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

        if args.multi_label:
            link_label = link_label.tolist()
            link_label = np.expand_dims(link_label, axis=0)
            link_label = torch.tensor(link_label, dtype=torch.long)
        print("link label:", link_label)

        # pred_label = self.pred_train[:, index]
        if args.multi_label:
            pred_label[pred_label < 0.5] = 0
            pred_label[pred_label >= 0.5] = 1
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
            ypred, src_adj_atts, dst_adj_atts = explainer(unconstrained=unconstrained)
            loss = explainer.loss(ypred, pred_label, epoch)
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
            # single_subgraph_label = sub_label.squeeze()

            if self.writer is not None:
                    self.writer.add_scalar("mask/src_density", src_mask_density, epoch)
                    self.writer.add_scalar("mask/dst_density", dst_mask_density, epoch)
                    self.writer.add_scalar(
                        "optimization/lr",
                        explainer.optimizer.param_groups[0]["lr"],
                        epoch,
                    )
                    # if epoch % 25 == 0:
                    #     explainer.log_mask(epoch)
                    #     explainer.log_masked_adj(
                    #         node_idx_new, epoch, label=single_subgraph_label
                    #     )
                    #     explainer.log_adj_grad(
                    #         node_idx_new, pred_label, epoch, label=single_subgraph_label
                    #     )

                    # if epoch == 0:
                    #     if self.model.att:
                    #         # explain node
                    #         print("adj att size: ", adj_atts.size())
                    #         adj_att = torch.sum(adj_atts[0], dim=2)
                    #         # adj_att = adj_att[neighbors][:, neighbors]
                    #         node_adj_att = adj_att * adj.float().cuda()
                    #         io_utils.log_matrix(
                    #             self.writer, node_adj_att[0], "att/matrix", epoch
                    #         )
                    #         node_adj_att = node_adj_att[0].cpu().detach().numpy()
                    #         G = io_utils.denoise_graph(
                    #             node_adj_att,
                    #             node_idx_new,
                    #             threshold=3.8,  # threshold_num=20,
                    #             max_component=True,
                    #         )
                    #         io_utils.log_graph(
                    #             self.writer,
                    #             G,
                    #             name="att/graph",
                    #             identify_self=not self.graph_mode,
                    #             nodecolor="label",
                    #             edge_vmax=None,
                    #             args=self.args,
                    #         )
            if model != "exp":
                break

        print("finished training in ", time.time() - begin_time)
        if model == "exp":
            src_masked_adj = (
                explainer.src_masked_adj[0].cpu().detach().numpy() * src_adj[0].cpu().detach().numpy()
            )
            dst_masked_adj = (
                explainer.dst_masked_adj[0].cpu().detach().numpy() * dst_adj[0].cpu().detach().numpy()
            )
            src_sub_feat = (
                explainer.src_feat_mask.cpu().detach().numpy() * src_x[0].cpu().detach().numpy()
            )
            dst_sub_feat = (
                explainer.dst_feat_mask.cpu().detach().numpy() * dst_x[0].cpu().detach().numpy()
            )
        # else:
        #     adj_atts = nn.functional.sigmoid(adj_atts).squeeze()
        #     masked_adj = adj_atts.cpu().detach().numpy() * sub_adj.squeeze()
        #
        fname = 'masked_adj_' + io_utils.gen_explainer_prefix(self.args) + (
                'src_idx_'+str(src_idx)+'dst_idx_'+str(dst_idx)+'graph_idx_'+str(self.graph_idx)+'.npy')
        with open(os.path.join(self.args.logdir, fname), 'wb') as outfile:
            np.save(outfile, np.asarray(src_masked_adj.copy()))
            np.save(outfile, np.asarray(dst_masked_adj.copy()))
            np.save(outfile, np.array(src_sub_feat.copy()))
            np.save(outfile, np.array(dst_sub_feat.copy()))
            print("Saved adjacency matrix to ", fname)

        src_results = {
            "src_sub_feat": src_sub_feat,
            "src_masked_adj": src_masked_adj,
            "src_idx_new": src_idx_new,
            "src_sub_label": src_sub_label,
            "src_neighbors": src_neighbors,
        }
        dst_results = {
            "dst_sub_feat": dst_sub_feat,
            "dst_masked_adj": dst_masked_adj,
            "dst_idx_new": dst_idx_new,
            "dst_sub_label": dst_sub_label,
            "dst_neighbors": dst_neighbors,
        }
        return src_results, dst_results

    # Main method
    def explain(
            self, node_idx, graph_idx=0, graph_mode=False, unconstrained=False, model="exp"
    ):
        """Explain a single node prediction
        """
        # index of the query node in the new adj
        if graph_mode:
            node_idx_new = node_idx
            sub_adj = self.adj[graph_idx]
            sub_feat = self.feat[graph_idx, :]
            sub_label = self.label[graph_idx]
            neighbors = np.asarray(range(self.adj.shape[0]))
        else:
            print("node label: ", self.label[graph_idx][node_idx])
            node_idx_new, sub_adj, sub_feat, sub_label, neighbors = self.extract_neighborhood(
                node_idx, graph_idx
            )
            print("neigh graph idx: ", node_idx, node_idx_new)
            sub_label = np.expand_dims(sub_label, axis=0)

        sub_adj = np.expand_dims(sub_adj, axis=0)
        sub_feat = np.expand_dims(sub_feat, axis=0)

        adj = torch.tensor(sub_adj, dtype=torch.float)
        x = torch.tensor(sub_feat, requires_grad=True, dtype=torch.float)
        label = torch.tensor(sub_label, dtype=torch.long)

        if self.graph_mode:
            pred_label = np.argmax(self.pred[0][graph_idx], axis=0)
            print("Graph predicted label: ", pred_label)
        else:
            pred_label = np.argmax(self.pred[graph_idx][neighbors], axis=1)
            print("Node predicted label: ", pred_label[node_idx_new])

        explainer = ExplainModule(
            adj=adj,
            x=x,
            model=self.model,
            label=label,
            args=self.args,
            writer=self.writer,
            graph_idx=self.graph_idx,
            graph_mode=self.graph_mode,
        )
        if self.args.gpu:
            explainer = explainer.cuda()

        self.model.eval()

        # gradient baseline
        if model == "grad":
            explainer.zero_grad()
            # pdb.set_trace()
            adj_grad = torch.abs(
                explainer.adj_feat_grad(node_idx_new, pred_label[node_idx_new])[0]
            )[graph_idx]
            masked_adj = adj_grad + adj_grad.t()
            masked_adj = nn.functional.sigmoid(masked_adj)
            masked_adj = masked_adj.cpu().detach().numpy() * sub_adj.squeeze()
        else:
            explainer.train()
            begin_time = time.time()
            for epoch in range(self.args.num_epochs):
                explainer.zero_grad()
                explainer.optimizer.zero_grad()
                ypred, adj_atts = explainer(node_idx_new, unconstrained=unconstrained)
                loss = explainer.loss(ypred, pred_label, node_idx_new, epoch)
                loss.backward()

                explainer.optimizer.step()
                if explainer.scheduler is not None:
                    explainer.scheduler.step()

                mask_density = explainer.mask_density()
                if self.print_training:
                    print(
                        "epoch: ",
                        epoch,
                        "; loss: ",
                        loss.item(),
                        "; mask density: ",
                        mask_density.item(),
                        "; pred: ",
                        ypred,
                    )
                single_subgraph_label = sub_label.squeeze()

                if self.writer is not None:
                    self.writer.add_scalar("mask/density", mask_density, epoch)
                    self.writer.add_scalar(
                        "optimization/lr",
                        explainer.optimizer.param_groups[0]["lr"],
                        epoch,
                    )
                    if epoch % 25 == 0:
                        explainer.log_mask(epoch)
                        explainer.log_masked_adj(
                            node_idx_new, epoch, label=single_subgraph_label
                        )
                        explainer.log_adj_grad(
                            node_idx_new, pred_label, epoch, label=single_subgraph_label
                        )

                    if epoch == 0:
                        if self.model.att:
                            # explain node
                            print("adj att size: ", adj_atts.size())
                            adj_att = torch.sum(adj_atts[0], dim=2)
                            # adj_att = adj_att[neighbors][:, neighbors]
                            node_adj_att = adj_att * adj.float().cuda()
                            io_utils.log_matrix(
                                self.writer, node_adj_att[0], "att/matrix", epoch
                            )
                            node_adj_att = node_adj_att[0].cpu().detach().numpy()
                            G = io_utils.denoise_graph(
                                node_adj_att,
                                node_idx_new,
                                threshold=3.8,  # threshold_num=20,
                                max_component=True,
                            )
                            io_utils.log_graph(
                                self.writer,
                                G,
                                name="att/graph",
                                identify_self=not self.graph_mode,
                                nodecolor="label",
                                edge_vmax=None,
                                args=self.args,
                            )
                if model != "exp":
                    break

            print("finished training in ", time.time() - begin_time)
            if model == "exp":
                masked_adj = (
                        explainer.masked_adj[0].cpu().detach().numpy() * sub_adj.squeeze()
                )
            else:
                adj_atts = nn.functional.sigmoid(adj_atts).squeeze()
                masked_adj = adj_atts.cpu().detach().numpy() * sub_adj.squeeze()

        fname = 'masked_adj_' + io_utils.gen_explainer_prefix(self.args) + (
                'node_idx_' + str(node_idx) + 'graph_idx_' + str(self.graph_idx) + '.npy')
        with open(os.path.join(self.args.logdir, fname), 'wb') as outfile:
            np.save(outfile, np.asarray(masked_adj.copy()))
            print("Saved adjacency matrix to ", fname)
        return masked_adj

    # NODE EXPLAINER
    def explain_nodes(self, node_indices, args, graph_idx=0):
        """
        Explain nodes

        Args:
            - node_indices  :  Indices of the nodes to be explained 
            - args          :  Program arguments (mainly for logging paths)
            - graph_idx     :  Index of the graph to explain the nodes from (if multiple).
        """
        masked_adjs = [
            self.explain(node_idx, graph_idx=graph_idx) for node_idx in node_indices
        ]
        ref_idx = node_indices[0]
        ref_adj = masked_adjs[0]
        curr_idx = node_indices[1]
        curr_adj = masked_adjs[1]
        new_ref_idx, _, ref_feat, _, _ = self.extract_neighborhood(ref_idx)
        new_curr_idx, _, curr_feat, _, _ = self.extract_neighborhood(curr_idx)

        G_ref = io_utils.denoise_graph(ref_adj, new_ref_idx, ref_feat, threshold=0.1)
        denoised_ref_feat = np.array(
            [G_ref.nodes[node]["feat"] for node in G_ref.nodes()]
        )
        denoised_ref_adj = nx.to_numpy_matrix(G_ref)
        # ref center node
        ref_node_idx = list(G_ref.nodes()).index(new_ref_idx)

        G_curr = io_utils.denoise_graph(
            curr_adj, new_curr_idx, curr_feat, threshold=0.1
        )
        denoised_curr_feat = np.array(
            [G_curr.nodes[node]["feat"] for node in G_curr.nodes()]
        )
        denoised_curr_adj = nx.to_numpy_matrix(G_curr)
        # curr center node
        curr_node_idx = list(G_curr.nodes()).index(new_curr_idx)

        P, aligned_adj, aligned_feat = self.align(
            denoised_ref_feat,
            denoised_ref_adj,
            ref_node_idx,
            denoised_curr_feat,
            denoised_curr_adj,
            curr_node_idx,
            args=args,
        )
        io_utils.log_matrix(self.writer, P, "align/P", 0)

        G_ref = nx.convert_node_labels_to_integers(G_ref)
        io_utils.log_graph(self.writer, G_ref, "align/ref")
        G_curr = nx.convert_node_labels_to_integers(G_curr)
        io_utils.log_graph(self.writer, G_curr, "align/before")

        P = P.cpu().detach().numpy()
        aligned_adj = aligned_adj.cpu().detach().numpy()
        aligned_feat = aligned_feat.cpu().detach().numpy()

        aligned_idx = np.argmax(P[:, curr_node_idx])
        print("aligned self: ", aligned_idx)
        G_aligned = io_utils.denoise_graph(
            aligned_adj, aligned_idx, aligned_feat, threshold=0.5
        )
        io_utils.log_graph(self.writer, G_aligned, "mask/aligned")

        # io_utils.log_graph(self.writer, aligned_adj.cpu().detach().numpy(), new_curr_idx,
        #        'align/aligned', epoch=1)

        return masked_adjs

    def explain_nodes_gnn_stats(self, node_indices, args, graph_idx=0, model="exp"):
        masked_adjs = [
            self.explain(node_idx, graph_idx=graph_idx, model=model)
            for node_idx in node_indices
        ]
        # pdb.set_trace()
        graphs = []
        feats = []
        adjs = []
        pred_all = []
        real_all = []
        for i, idx in enumerate(node_indices):
            new_idx, _, feat, _, _ = self.extract_neighborhood(idx)
            G = io_utils.denoise_graph(masked_adjs[i], new_idx, feat, threshold_num=20)
            pred, real = self.make_pred_real(masked_adjs[i], new_idx)
            pred_all.append(pred)
            real_all.append(real)
            denoised_feat = np.array([G.nodes[node]["feat"] for node in G.nodes()])
            denoised_adj = nx.to_numpy_matrix(G)
            graphs.append(G)
            feats.append(denoised_feat)
            adjs.append(denoised_adj)
            io_utils.log_graph(
                self.writer,
                G,
                "graph/{}_{}_{}".format(self.args.dataset, model, i),
                identify_self=True,
            )

        pred_all = np.concatenate((pred_all), axis=0)
        real_all = np.concatenate((real_all), axis=0)

        auc_all = roc_auc_score(real_all, pred_all)
        precision, recall, thresholds = precision_recall_curve(real_all, pred_all)

        plt.switch_backend("agg")
        plt.plot(recall, precision)
        plt.savefig("log/pr/pr_" + self.args.dataset + "_" + model + ".png")

        plt.close()

        auc_all = roc_auc_score(real_all, pred_all)
        precision, recall, thresholds = precision_recall_curve(real_all, pred_all)

        plt.switch_backend("agg")
        plt.plot(recall, precision)
        plt.savefig("log/pr/pr_" + self.args.dataset + "_" + model + ".png")

        plt.close()

        with open("log/pr/auc_" + self.args.dataset + "_" + model + ".txt", "w") as f:
            f.write(
                "dataset: {}, model: {}, auc: {}\n".format(
                    self.args.dataset, "exp", str(auc_all)
                )
            )

        return masked_adjs

    # GRAPH EXPLAINER
    def explain_graphs(self, graph_indices):
        """
        Explain graphs.
        """
        masked_adjs = []

        for graph_idx in graph_indices:
            masked_adj = self.explain(node_idx=0, graph_idx=graph_idx, graph_mode=True)
            G_denoised = io_utils.denoise_graph(
                masked_adj,
                0,
                threshold_num=20,
                feat=self.feat[graph_idx],
                max_component=False,
            )
            label = self.label[graph_idx]
            io_utils.log_graph(
                self.writer,
                G_denoised,
                "graph/graphidx_{}_label={}".format(graph_idx, label),
                identify_self=False,
                nodecolor="feat",
            )
            masked_adjs.append(masked_adj)

            G_orig = io_utils.denoise_graph(
                self.adj[graph_idx],
                0,
                feat=self.feat[graph_idx],
                threshold=None,
                max_component=False,
            )

            io_utils.log_graph(
                self.writer,
                G_orig,
                "graph/graphidx_{}".format(graph_idx),
                identify_self=False,
                nodecolor="feat",
            )

        # plot cmap for graphs' node features
        io_utils.plot_cmap_tb(self.writer, "tab20", 20, "tab20_cmap")

        return masked_adjs

    def log_representer(self, rep_val, sim_val, alpha, graph_idx=0):
        """ visualize output of representer instances. """
        rep_val = rep_val.cpu().detach().numpy()
        sim_val = sim_val.cpu().detach().numpy()
        alpha = alpha.cpu().detach().numpy()
        sorted_rep = sorted(range(len(rep_val)), key=lambda k: rep_val[k])
        print(sorted_rep)
        topk = 5
        most_neg_idx = [sorted_rep[i] for i in range(topk)]
        most_pos_idx = [sorted_rep[-i - 1] for i in range(topk)]
        rep_idx = [most_pos_idx, most_neg_idx]

        if self.graph_mode:
            pred = np.argmax(self.pred[0][graph_idx], axis=0)
        else:
            pred = np.argmax(self.pred[graph_idx][self.train_idx], axis=1)
        print(metrics.confusion_matrix(self.label[graph_idx][self.train_idx], pred))
        plt.switch_backend("agg")
        fig = plt.figure(figsize=(5, 3), dpi=600)
        for i in range(2):
            for j in range(topk):
                idx = self.train_idx[rep_idx[i][j]]
                print(
                    "node idx: ",
                    idx,
                    "; node label: ",
                    self.label[graph_idx][idx],
                    "; pred: ",
                    pred,
                )

                idx_new, sub_adj, sub_feat, sub_label, neighbors = self.extract_neighborhood(
                    idx, graph_idx
                )
                G = nx.from_numpy_matrix(sub_adj)
                node_colors = [1 for i in range(G.number_of_nodes())]
                node_colors[idx_new] = 0
                # node_color='#336699',

                ax = plt.subplot(2, topk, i * topk + j + 1)
                nx.draw(
                    G,
                    pos=nx.spring_layout(G),
                    with_labels=True,
                    font_size=4,
                    node_color=node_colors,
                    cmap=plt.get_cmap("Set1"),
                    vmin=0,
                    vmax=8,
                    edge_vmin=0.0,
                    edge_vmax=1.0,
                    width=0.5,
                    node_size=25,
                    alpha=0.7,
                )
                ax.xaxis.set_visible(False)
        fig.canvas.draw()
        self.writer.add_image(
            "local/representer_neigh", tensorboardX.utils.figure_to_image(fig), 0
        )

    def representer(self):
        """
        experiment using representer theorem for finding supporting instances.
        https://papers.nips.cc/paper/8141-representer-point-selection-for-explaining-deep-neural-networks.pdf
        """
        self.model.train()
        self.model.zero_grad()
        adj = torch.tensor(self.adj, dtype=torch.float)
        x = torch.tensor(self.feat, requires_grad=True, dtype=torch.float)
        label = torch.tensor(self.label, dtype=torch.long)
        if self.args.gpu:
            adj, x, label = adj.cuda(), x.cuda(), label.cuda()

        preds, _ = self.model(x, adj)
        preds.retain_grad()
        self.embedding = self.model.embedding_tensor
        loss = self.model.loss(preds, label)
        loss.backward()
        self.preds_grad = preds.grad
        pred_idx = np.expand_dims(np.argmax(self.pred, axis=2), axis=2)
        pred_idx = torch.LongTensor(pred_idx)
        if self.args.gpu:
            pred_idx = pred_idx.cuda()
        self.alpha = self.preds_grad


    # Utilities
    def extract_neighborhood(self, node_idx, graph_idx=0):
        """Returns the neighborhood of a given ndoe."""
        neighbors_adj_row = self.neighborhoods[graph_idx][node_idx, :]
        # index of the query node in the new adj
        node_idx_new = sum(neighbors_adj_row[:node_idx])
        neighbors = np.nonzero(neighbors_adj_row)[0]
        sub_adj = self.adj[graph_idx][neighbors][:, neighbors]
        sub_feat = self.feat[graph_idx, neighbors]
        sub_label = self.label[graph_idx][neighbors]
        return node_idx_new, sub_adj, sub_feat, sub_label, neighbors

    def align(
        self, ref_feat, ref_adj, ref_node_idx, curr_feat, curr_adj, curr_node_idx, args
    ):
        """ Tries to find an alignment between two graphs. 
        """
        ref_adj = torch.FloatTensor(ref_adj)
        curr_adj = torch.FloatTensor(curr_adj)

        ref_feat = torch.FloatTensor(ref_feat)
        curr_feat = torch.FloatTensor(curr_feat)

        P = nn.Parameter(torch.FloatTensor(ref_adj.shape[0], curr_adj.shape[0]))
        with torch.no_grad():
            nn.init.constant_(P, 1.0 / ref_adj.shape[0])
            P[ref_node_idx, :] = 0.0
            P[:, curr_node_idx] = 0.0
            P[ref_node_idx, curr_node_idx] = 1.0
        opt = torch.optim.Adam([P], lr=0.01, betas=(0.5, 0.999))
        for i in range(args.align_steps):
            opt.zero_grad()
            feat_loss = torch.norm(P @ curr_feat - ref_feat)

            aligned_adj = P @ curr_adj @ torch.transpose(P, 0, 1)
            align_loss = torch.norm(aligned_adj - ref_adj)
            loss = feat_loss + align_loss
            loss.backward()  # Calculate gradients
            self.writer.add_scalar("optimization/align_loss", loss, i)
            print("iter: ", i, "; loss: ", loss)
            opt.step()

        return P, aligned_adj, P @ curr_feat

    def make_pred_real(self, adj, start):
        # house graph
        if self.args.dataset == "syn1" or self.args.dataset == "syn2":
            # num_pred = max(G.number_of_edges(), 6)
            pred = adj[np.triu(adj) > 0]
            real = adj.copy()

            if real[start][start + 1] > 0:
                real[start][start + 1] = 10
            if real[start + 1][start + 2] > 0:
                real[start + 1][start + 2] = 10
            if real[start + 2][start + 3] > 0:
                real[start + 2][start + 3] = 10
            if real[start][start + 3] > 0:
                real[start][start + 3] = 10
            if real[start][start + 4] > 0:
                real[start][start + 4] = 10
            if real[start + 1][start + 4]:
                real[start + 1][start + 4] = 10
            real = real[np.triu(real) > 0]
            real[real != 10] = 0
            real[real == 10] = 1

        # cycle graph
        elif self.args.dataset == "syn4":
            pred = adj[np.triu(adj) > 0]
            real = adj.copy()
            # pdb.set_trace()
            if real[start][start + 1] > 0:
                real[start][start + 1] = 10
            if real[start + 1][start + 2] > 0:
                real[start + 1][start + 2] = 10
            if real[start + 2][start + 3] > 0:
                real[start + 2][start + 3] = 10
            if real[start + 3][start + 4] > 0:
                real[start + 3][start + 4] = 10
            if real[start + 4][start + 5] > 0:
                real[start + 4][start + 5] = 10
            if real[start][start + 5]:
                real[start][start + 5] = 10
            real = real[np.triu(real) > 0]
            real[real != 10] = 0
            real[real == 10] = 1

        return pred, real


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
            "size": 0.005,
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
                    src_x = src_x * src_feat_mask
                    dst_x = dst_x * dst_feat_mask

        ypred, src_adj_att, dst_adj_att = self.model(src_x, self.src_masked_adj, [self.src_idx_new, self.dst_idx_new], dst_x, self.dst_masked_adj)
        # if self.graph_mode:
        #     res = nn.Softmax(dim=0)(ypred[0])
        # else:
        #     node_pred = ypred[self.graph_idx, node_idx, :]
        #     res = nn.Softmax(dim=0)(node_pred)
        return ypred, src_adj_att, dst_adj_att

    def adj_feat_grad(self, node_idx, pred_label_node):
        self.model.zero_grad()
        self.adj.requires_grad = True
        self.x.requires_grad = True
        if self.adj.grad is not None:
            self.adj.grad.zero_()
            self.x.grad.zero_()
        if self.args.gpu:
            adj = self.adj.cuda()
            x = self.x.cuda()
            label = self.label.cuda()
        else:
            x, adj = self.x, self.adj
        ypred, _ = self.model(x, adj)
        if self.graph_mode:
            logit = nn.Softmax(dim=0)(ypred[0])
        else:
            logit = nn.Softmax(dim=0)(ypred[self.graph_idx, node_idx, :])
        logit = logit[pred_label_node]
        loss = -torch.log(logit)
        loss.backward()
        return self.adj.grad, self.x.grad

    def loss(self, pred, pred_label, epoch):
        """
        Args:
            pred: prediction made by current model
            pred_label: the label predicted by the original model.
        """
        mi_obj = False
        if mi_obj:
            pred_loss = -torch.sum(pred * torch.log(pred))
        else:
            # pred_label_node = pred_label if self.graph_mode else pred_label[node_idx]
            # gt_label_node = self.label if self.graph_mode else self.label[0][node_idx]
            # logit = pred[gt_label_node]
            # pred_loss = -torch.log(logit)
            pred_loss = torch.nn.functional.binary_cross_entropy(torch.sigmoid(pred), torch.sigmoid(pred_label.float()))
        # size
        src_mask = self.src_mask
        dst_mask = self.dst_mask
        if self.mask_act == "sigmoid":
            src_mask = torch.sigmoid(self.src_mask)
            dst_mask = torch.sigmoid(self.dst_mask)
        elif self.mask_act == "ReLU":
            src_mask = nn.ReLU()(self.src_mask)
            dst_mask = nn.ReLU()(self.dst_mask)
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

        loss = pred_loss + size_loss + lap_loss + mask_ent_loss + feat_size_loss
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

    def log_mask(self, epoch):
        plt.switch_backend("agg")
        fig = plt.figure(figsize=(4, 3), dpi=400)
        plt.imshow(self.mask.cpu().detach().numpy(), cmap=plt.get_cmap("BuPu"))
        cbar = plt.colorbar()
        cbar.solids.set_edgecolor("face")

        plt.tight_layout()
        fig.canvas.draw()
        self.writer.add_image(
            "mask/mask", tensorboardX.utils.figure_to_image(fig), epoch
        )

        # fig = plt.figure(figsize=(4,3), dpi=400)
        # plt.imshow(self.feat_mask.cpu().detach().numpy()[:,np.newaxis], cmap=plt.get_cmap('BuPu'))
        # cbar = plt.colorbar()
        # cbar.solids.set_edgecolor("face")

        # plt.tight_layout()
        # fig.canvas.draw()
        # self.writer.add_image('mask/feat_mask', tensorboardX.utils.figure_to_image(fig), epoch)
        io_utils.log_matrix(
            self.writer, torch.sigmoid(self.feat_mask), "mask/feat_mask", epoch
        )

        fig = plt.figure(figsize=(4, 3), dpi=400)
        # use [0] to remove the batch dim
        plt.imshow(self.masked_adj[0].cpu().detach().numpy(), cmap=plt.get_cmap("BuPu"))
        cbar = plt.colorbar()
        cbar.solids.set_edgecolor("face")

        plt.tight_layout()
        fig.canvas.draw()
        self.writer.add_image(
            "mask/adj", tensorboardX.utils.figure_to_image(fig), epoch
        )

        if self.args.mask_bias:
            fig = plt.figure(figsize=(4, 3), dpi=400)
            # use [0] to remove the batch dim
            plt.imshow(self.mask_bias.cpu().detach().numpy(), cmap=plt.get_cmap("BuPu"))
            cbar = plt.colorbar()
            cbar.solids.set_edgecolor("face")

            plt.tight_layout()
            fig.canvas.draw()
            self.writer.add_image(
                "mask/bias", tensorboardX.utils.figure_to_image(fig), epoch
            )

    def log_adj_grad(self, node_idx, pred_label, epoch, label=None):
        log_adj = False

        if self.graph_mode:
            predicted_label = pred_label
            # adj_grad, x_grad = torch.abs(self.adj_feat_grad(node_idx, predicted_label)[0])[0]
            adj_grad, x_grad = self.adj_feat_grad(node_idx, predicted_label)
            adj_grad = torch.abs(adj_grad)[0]
            x_grad = torch.sum(x_grad[0], 0, keepdim=True).t()
        else:
            predicted_label = pred_label[node_idx]
            # adj_grad = torch.abs(self.adj_feat_grad(node_idx, predicted_label)[0])[self.graph_idx]
            adj_grad, x_grad = self.adj_feat_grad(node_idx, predicted_label)
            adj_grad = torch.abs(adj_grad)[self.graph_idx]
            x_grad = x_grad[self.graph_idx][node_idx][:, np.newaxis]
            # x_grad = torch.sum(x_grad[self.graph_idx], 0, keepdim=True).t()
        adj_grad = (adj_grad + adj_grad.t()) / 2
        adj_grad = (adj_grad * self.adj).squeeze()
        if log_adj:
            io_utils.log_matrix(self.writer, adj_grad, "grad/adj_masked", epoch)
            self.adj.requires_grad = False
            io_utils.log_matrix(self.writer, self.adj.squeeze(), "grad/adj_orig", epoch)

        masked_adj = self.masked_adj[0].cpu().detach().numpy()

        # only for graph mode since many node neighborhoods for syn tasks are relatively large for
        # visualization
        if self.graph_mode:
            G = io_utils.denoise_graph(
                masked_adj, node_idx, feat=self.x[0], threshold=None, max_component=False
            )
            io_utils.log_graph(
                self.writer,
                G,
                name="grad/graph_orig",
                epoch=epoch,
                identify_self=False,
                label_node_feat=True,
                nodecolor="feat",
                edge_vmax=None,
                args=self.args,
            )
        io_utils.log_matrix(self.writer, x_grad, "grad/feat", epoch)

        adj_grad = adj_grad.detach().numpy()
        if self.graph_mode:
            print("GRAPH model")
            G = io_utils.denoise_graph(
                adj_grad,
                node_idx,
                feat=self.x[0],
                threshold=0.0003,  # threshold_num=20,
                max_component=True,
            )
            io_utils.log_graph(
                self.writer,
                G,
                name="grad/graph",
                epoch=epoch,
                identify_self=False,
                label_node_feat=True,
                nodecolor="feat",
                edge_vmax=None,
                args=self.args,
            )
        else:
            # G = io_utils.denoise_graph(adj_grad, node_idx, label=label, threshold=0.5)
            G = io_utils.denoise_graph(adj_grad, node_idx, threshold_num=12)
            io_utils.log_graph(
                self.writer, G, name="grad/graph", epoch=epoch, args=self.args
            )

        # if graph attention, also visualize att

    def log_masked_adj(self, node_idx, epoch, name="mask/graph", label=None):
        # use [0] to remove the batch dim
        masked_adj = self.masked_adj[0].cpu().detach().numpy()
        if self.graph_mode:
            G = io_utils.denoise_graph(
                masked_adj,
                node_idx,
                feat=self.x[0],
                threshold=0.2,  # threshold_num=20,
                max_component=True,
            )
            io_utils.log_graph(
                self.writer,
                G,
                name=name,
                identify_self=False,
                nodecolor="feat",
                epoch=epoch,
                label_node_feat=True,
                edge_vmax=None,
                args=self.args,
            )
        else:
            G = io_utils.denoise_graph(
                masked_adj, node_idx, threshold_num=12, max_component=True
            )
            io_utils.log_graph(
                self.writer,
                G,
                name=name,
                identify_self=True,
                nodecolor="label",
                epoch=epoch,
                edge_vmax=None,
                args=self.args,
            )

