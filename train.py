""" train.py

    Main interface to train the GNNs that will be later explained.
"""
import argparse
import os
import pickle
import random
import shutil
import time

import matplotlib
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

import networkx as nx
import numpy as np
import sklearn.metrics as metrics

import torch
import torch.nn as nn
from torch.autograd import Variable

from tensorboardX import SummaryWriter

import configs
import gengraph

import utils.math_utils as math_utils
import utils.io_utils as io_utils
import utils.parser_utils as parser_utils
import utils.train_utils as train_utils
import utils.featgen as featgen
import utils.graph_utils as graph_utils

import models


#############################
#
# Prepare Data
#
#############################
def prepare_data(graph, edge_labels, args, test_graphs=None):
    edges = list(graph.edges())
    num_edges = len(edges)
    num_train = int(num_edges * args.fraction * args.train_ratio)
    num_test = int(num_edges * args.fraction * (1 - args.train_ratio))
    num_edge_labels = max(edge_labels) + 1

    # generate negative data
    # num_nodes = graph.number_of_nodes()
    # neg_data = []
    # neg_labels = []
    # k = 0
    # total_num = num_train + num_test
    # while (k < total_num):
    #     src = random.randint(0, num_nodes-1)
    #     dst = random.randint(0, num_nodes-1)
    #     elabel = random.randint(0, num_edge_labels-1)
    #     if not graph.has_edge(src, dst):
    #         neg_data.append((src, dst))
    #         neg_labels.append(elabel)
    #         k = k + 1
    #     # else:
    #     #     flag = 0
    #     #     for i in range(len(graph[src][dst])):
    #     #         if graph[src][dst][i]['label'] == elabel:
    #     #             flag = 1
    #     #             break
    #     #     if flag == 0:
    #     #         neg_data.append((src, dst))
    #     #         neg_labels.append(elabel)
    #     #         k = k + 1
    # neg_train_edges = neg_data[:num_train]
    # neg_train_labels = neg_labels[:num_train]
    # neg_test_edges = neg_data[num_train:]
    # neg_test_labels = neg_labels[num_train:]

    # generate positive data
    idx = [i for i in range(num_edges)]
    np.random.shuffle(idx)
    pos_idx = idx[:(num_train+num_test)]
    pos_edges = []
    pos_labels = []
    for index in pos_idx:
        src = edges[index][0]
        dst = edges[index][1]
        if not graph.has_edge(src, dst):
            continue
        if len(graph[src][dst]) == 1:
            pos_edges.append(edges[index])
            pos_labels.append(edge_labels[index])
            graph.remove_edge(src, dst)
        else:
            pos_edges.append(edges[index])
            pos_labels.append([])
            for i in reversed(range(len(graph[src][dst]))):
                pos_labels[len(pos_labels) - 1].append(graph[src][dst][i]['label'])
                graph.remove_edge(src, dst)       # in a pop way

    # change edge labels into one-hop form
    for i in range(len(pos_labels)):
        edge_label_one_hot = [0] * num_edge_labels
        edge_labels = pos_labels[i]
        if isinstance(edge_labels, int):
            edge_label_one_hot[edge_labels] = 1
        else:
            for label in edge_labels:
                edge_label_one_hot[label] = 1
        pos_labels[i] = edge_label_one_hot

    num_train = int(len(pos_edges) * args.train_ratio)
    pos_train_edges = pos_edges[:num_train]
    pos_test_edges = pos_edges[num_train:]
    pos_train_labels = pos_labels[:num_train]
    pos_test_labels = pos_labels[num_train:]
    print(
        "Num training edges: ",
        len(pos_train_edges),
        # len(pos_train_edges) + len(neg_train_edges),
        "; Num testing edges: ",
        len(pos_test_edges),
        # len(pos_test_edges) + len(neg_test_edges)
    )

    print("Number of edges left: ", graph.number_of_edges())

    # for i in range(len(pos_test_labels)):
    #     edge_label_one_hot = [0] * num_edge_labels
    #     edge_label = pos_test_labels[i]
    #     edge_label_one_hot[edge_label] = 1
    #     pos_test_labels[i] = edge_label_one_hot

    # for i in range(len(neg_train_labels)):
    #     edge_label_one_hot = [0] * num_edge_labels
    #     # edge_label = neg_train_labels[i]
    #     # edge_label_one_hot[edge_label] = 1
    #     neg_train_labels[i] = edge_label_one_hot
    #
    # for i in range(len(neg_test_labels)):
    #     edge_label_one_hot = [0] * num_edge_labels
    #     # edge_label = neg_test_labels[i]
    #     # edge_label_one_hot[edge_label] = 1
    #     neg_test_labels[i] = edge_label_one_hot

    train_data = pos_train_edges # + neg_train_edges
    train_labels = pos_train_labels # + neg_train_labels

    test_data = pos_test_edges # + neg_test_edges
    test_labels = pos_test_labels # + neg_test_labels

    return graph, np.array(train_data), np.array(train_labels), np.array(test_data), np.array(test_labels)


#############################
#
# Training 
#
#############################

def train_link_classifier(G, node_labels, train_data, train_labels, test_data, test_labels, model, args, writer=None):
    # for p in model.parameters():
    #     if p.grad is not None:
    #         print(p.grad.data)

    # adj = nx.to_numpy_array(G)   # wrong
    num_nodes = G.number_of_nodes()
    adj_origin = [[0] * num_nodes] * num_nodes
    adj_origin = np.array(adj_origin)

    edges = list(G.edges())
    for edge in edges:
        adj_origin[edge[0]][edge[1]] += 1
    adj = adj_origin.transpose()   # for collecting information from in-edges during 'forward' in GraphConv class.

    existing_node = list(G.nodes)[-1]
    feat_dim = G.nodes[existing_node]["feat"].shape[0]
    x = np.zeros((num_nodes, feat_dim), dtype=float)
    for i, u in enumerate(G.nodes()):
        x[i, :] = G.nodes[u]["feat"]

    adj = np.expand_dims(adj, axis=0)
    x = np.expand_dims(x, axis=0)
    train_labels = np.expand_dims(train_labels, axis=0)

    adj = torch.tensor(adj, dtype=torch.float)
    x = torch.tensor(x, requires_grad=True, dtype=torch.float)
    train_labels = torch.tensor(train_labels, dtype=torch.long)

    scheduler, optimizer = train_utils.build_optimizer(
        args, model.parameters(), weight_decay=args.weight_decay
    )
    model.train()
    ypred_train = None
    for epoch in range(args.num_epochs):
        begin_time = time.time()

        model.zero_grad()

        ypred_train, adj_att = model(x, adj, train_data)
        loss = model.loss(ypred_train, train_labels)

        loss.backward()
        # for param in model.parameters():
        #     print(param.grad)

        nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        optimizer.step()

        elapsed = time.time() - begin_time

        result_train, result_test = evaluate_node(model, adj, x, test_data, test_labels, ypred_train.cpu(), train_labels)

        if writer is not None:
            writer.add_scalar("loss/avg_loss", loss.item(), epoch)
            writer.add_scalars(
                "prec",
                {"train": result_train["prec"], "test": result_test["prec"]},
                epoch,
            )
            writer.add_scalars(
                "recall",
                {"train": result_train["recall"], "test": result_test["recall"]},
                epoch,
            )
            writer.add_scalars(
                "acc", {"train": result_train["acc"], "test": result_test["acc"]}, epoch
            )
        if epoch % 10 == 0:
            print(
                "epoch: ",
                epoch,
                "; loss: ",
                loss.item(),
                "; train_acc: ",
                result_train["acc"],
                "; test_acc: ",
                result_test["acc"],
                "; train_prec: ",
                result_train["prec"],
                "; test_prec: ",
                result_test["prec"],
                "; epoch time: ",
                "{0:0.2f}".format(elapsed),
            )

        if scheduler is not None:
            scheduler.step()

    # computation graph
    model.eval()
    ypred_train, _ = model(x, adj, train_data)
    ypred_test, _ = model(x, adj, test_data)
    cg_data = {
        "adj": adj.numpy(),
        "feat": x.detach().numpy(),
        "node_labels": np.expand_dims(node_labels, axis=0),
        "label_train": train_labels.numpy(),
        "label_test": np.expand_dims(test_labels, axis=0),
        "pred_train": ypred_train.cpu().detach().numpy(),
        "pred_test": ypred_test.cpu().detach().numpy(),
        "train_idx": train_data,
        "test_idx": test_data,
    }
    io_utils.save_checkpoint(model, optimizer, args, num_epochs=-1, cg_dict=cg_data)


#############################
#
# Evaluate Trained Model
#
#############################
def evaluate_node(model, adj, x, test_data, test_labels, ypred_train, train_labels):
    test_labels = np.expand_dims(test_labels, axis=0)
    test_labels = torch.tensor(test_labels, dtype=torch.long)
    ypred_test, adj_att = model(x, adj, test_data)

    ypred_train = ypred_train.detach().numpy()[0]
    ypred_train[ypred_train < 0.5] = 0
    ypred_train[ypred_train >= 0.5] = 1
    ypred_train = ypred_train.astype(int)

    ypred_test = ypred_test.detach().numpy()[0]
    ypred_test[ypred_test < 0.5] = 0
    ypred_test[ypred_test >= 0.5] = 1
    ypred_test = ypred_test.astype(int)

    train_labels = train_labels.numpy()[0]
    test_labels = test_labels.numpy()[0]

    result_train = {
        "prec": metrics.precision_score(train_labels, ypred_train, average="samples"),
        "recall": metrics.recall_score(train_labels, ypred_train, average="samples"),
        "acc": metrics.accuracy_score(train_labels, ypred_train),
    }
    result_test = {
        "prec": metrics.precision_score(test_labels, ypred_test, average="samples"),
        "recall": metrics.recall_score(test_labels, ypred_test, average="samples"),
        "acc": metrics.accuracy_score(test_labels, ypred_test),
    }
    return result_train, result_test


def link_prediction_task(args, writer=None):
    graph, node_labels, edge_labels, num_node_labels, num_edge_labels = io_utils.read_graphfile(
        args.datadir, args.dataset
    )
    input_dim = graph.graph["feat_dim"]

    graph, train_data, train_labels, test_data, test_labels = prepare_data(graph, edge_labels, args)

    model = models.GcnEncoderNode(
        input_dim,
        args.hidden_dim,
        args.output_dim,
        num_edge_labels,
        args.num_gc_layers,
        bn=args.bn,
        dropout=args.dropout,
        args=args,
    )

    train_link_classifier(graph, node_labels, train_data, train_labels, test_data, test_labels, model, args, writer=writer)


def main():
    prog_args = configs.arg_parse()

    path = os.path.join(prog_args.logdir, io_utils.gen_prefix(prog_args))
    writer = SummaryWriter(path)

    # io_utils.read_train_test_file(prog_args.datadir, prog_args.bmname)

    if prog_args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = prog_args.cuda
        print("CUDA", prog_args.cuda)
    else:
        print("Using CPU")

    # use --bmname=[dataset_name] for Reddit-Binary, Mutagenicity
    if prog_args.link_prediction is True:
        link_prediction_task(prog_args, writer=writer)

    writer.close()


if __name__ == "__main__":
    main()

