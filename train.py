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
import copy
import sys

import models

import ogb_models

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


#############################
#
# Prepare Data
#
#############################
def prepare_data(graph, num_edge_labels, args):
    edges = list(graph.edges())
    num_edges = len(edges)
    num_nodes = graph.number_of_nodes()
    num_train = int(num_edges * args.fraction * args.train_ratio)
    num_test = int(num_edges * args.fraction * (1 - args.train_ratio))

    if args.single_edge_label:      # only one type of edges in the graph
        # generate positive data
        idx = [i for i in range(num_edges)]
        np.random.shuffle(idx)
        pos_idx = idx[:(num_train + num_test)]
        pos_edges = []
        pos_labels = []
        for index in pos_idx:
            src = edges[index][0]
            dst = edges[index][1]
            if not graph.has_edge(src, dst):
                continue
            pos_edges.append(edges[index])
            pos_labels.append(1)
            graph.remove_edge(src, dst)
        num_train = int(len(pos_edges) * args.train_ratio)
        pos_train_edges = pos_edges[:num_train]
        pos_test_edges = pos_edges[num_train:]
        pos_train_labels = pos_labels[:num_train]
        pos_test_labels = pos_labels[num_train:]

        # generate negative data
        neg_data = []
        neg_labels = []
        k = 0
        while (k < len(pos_edges)):
            src = random.randint(0, num_nodes-1)
            dst = random.randint(0, num_nodes-1)
            if not graph.has_edge(src, dst):
                neg_data.append((src, dst))
                neg_labels.append(0)
                k = k + 1
        neg_train_edges = neg_data[:num_train]
        neg_train_labels = neg_labels[:num_train]
        neg_test_edges = neg_data[num_train:]
        neg_test_labels = neg_labels[num_train:]

        print(
            "Num training edges: ",
            len(pos_train_edges) + len(neg_train_edges),
            "; Num testing edges: ",
            len(pos_test_edges) + len(neg_test_edges)
        )
        print("Number of edges left: ", graph.number_of_edges())

        train_data = pos_train_edges + neg_train_edges
        train_labels = pos_train_labels + neg_train_labels
        test_data = pos_test_edges + neg_test_edges
        test_labels = pos_test_labels + neg_test_labels

        return graph, np.array(train_data), np.array(train_labels), np.array(test_data), np.array(test_labels)

    elif args.multi_label or args.multi_class:  # multi-type of edges in the graph
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
                pos_labels.append(graph[src][dst][0]["label"])
                graph.remove_edge(src, dst)
            else:
                pos_edges.append(edges[index])
                pos_labels.append([])
                for i in reversed(range(len(graph[src][dst]))):
                    pos_labels[len(pos_labels) - 1].append(graph[src][dst][i]['label'])
                    graph.remove_edge(src, dst)       # in a pop way

        if args.multi_label:
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
            "Num pos training edges: ",
            len(pos_train_edges),
            "; Num pos testing edges: ",
            len(pos_test_edges),
        )

        print("Number of edges left in Graph: ", graph.number_of_edges())

        train_data = pos_train_edges  # + neg_train_edges
        train_labels = pos_train_labels  # + neg_train_labels

        test_data = pos_test_edges  # + neg_test_edges
        test_labels = pos_test_labels  # + neg_test_labels

        return graph, np.array(train_data), np.array(train_labels), np.array(test_data), np.array(test_labels)


def generate_predict_data(graph, num_edge_labels, args):
    predict_data = []
    num_nodes = graph.number_of_nodes()
    if args.single_edge_label or args.multi_class:
        for src in range(num_nodes):
            for dst in range(num_nodes):
                if not graph.has_edge(src, dst):
                    predict_data.append((src, dst))
    elif args.multi_label:
        for src in range(num_nodes):
            for dst in range(num_nodes):
                # predict_data.append((src, dst))
                if not graph.has_edge(src, dst):
                    for elabel in range(num_edge_labels):
                        predict_data.append((src, dst))
                else:
                    for elabel in range(num_edge_labels):
                        if not graph.has_edge(src, dst, elabel):
                            predict_data.append((src, dst))
                            break
                        # flag = 0
                        # for i in range(len(graph[src][dst])):
                        #     if graph[src][dst][i]['label'] == elabel:
                        #         flag = 1
                        #         break
                        # if flag == 0:
                        #     predict_data.append((src, dst))
                        #     break
    print("generate ", len(predict_data), "predict data")
    return np.array(predict_data)


#############################
#
# Training 
#
#############################

def train_link_classifier(G, node_labels, train_data, train_labels, test_data, test_labels, model, args, writer=None):

    num_nodes = G.number_of_nodes()

    # adj_origin = nx.adjacency_matrix(G)
    adj_origin = nx.to_scipy_sparse_matrix(G)
    # adj_origin = [[0] * num_nodes] * num_nodes
    # adj_origin = np.array(adj_origin)
    # edges = list(G.edges())
    # for edge in edges:
    #     adj_origin[edge[0]][edge[1]] += 1
    adj = adj_origin.transpose()
    # Transpose to collect information from in-edges during 'forward' in GraphConv class when graph is directed.
    # For undirected graph, transpose or not transpose do not change anything.

    existing_node = list(G.nodes)[-1]
    feat_dim = G.nodes[existing_node]["feat"].shape[0]
    x = np.zeros((num_nodes, feat_dim), dtype=float)
    for i, u in enumerate(G.nodes()):
        x[i, :] = G.nodes[u]["feat"]

    if 'ogb' not in args.model:
        adj_origin = nx.to_numpy_array(G)
        adj = adj_origin.transpose()
        adj = np.expand_dims(adj, axis=0)
        adj = torch.tensor(adj, dtype=torch.float)
        x = np.expand_dims(x, axis=0)
    else:
        adj = torch.tensor(list(G.to_undirected().edges))
        #adj = np.expand_dims(adj, axis=0)

    train_labels = np.expand_dims(train_labels, axis=0)

    x = torch.tensor(x, requires_grad=True, dtype=torch.float)
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    train_data = torch.tensor(train_data, dtype=torch.long)
    test_data = torch.tensor(test_data, dtype=torch.long)

    scheduler, optimizer = train_utils.build_optimizer(
        args, model.parameters(), weight_decay=args.weight_decay
    )

    model.train()
    if args.model == "ogb_GCN":
        model.reset_parameters()
    ypred_train = None

    for epoch in range(args.num_epochs):
        begin_time = time.time()

        model.zero_grad()

        if args.gpu:
            ypred_train = model(x.cuda(), adj.cuda(), train_data.cuda())
        else:
            ypred_train = model(x, adj, train_data)

        if args.gpu:
            loss = model.loss(ypred_train, train_labels.cuda())
        else:
            loss = model.loss(ypred_train, train_labels)

        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        optimizer.step()

        elapsed = time.time() - begin_time

        start_time = time.time()
        result_train, result_test = evaluate_link(model, adj, x, test_data, test_labels, ypred_train.cpu(), train_labels.cpu(), args)
        end_time = time.time()
        if epoch % 10 == 0:
            print("evalate time: ", (end_time - start_time))

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
            if args.single_edge_label or args.multi_class:
                writer.add_scalars(
                    "acc", {"train": result_train["acc"], "test": result_test["acc"]}, epoch
                )
            elif args.multi_label:
                writer.add_scalars(
                    "F1", {"train": result_train["F1"], "test": result_test["F1"]}, epoch
                )
        if (args.single_edge_label or args.multi_class) and epoch % 10 == 0:
            print(
                "epoch: ",
                epoch,
                "; loss: ",
                loss.item(),
                "; train_prec: ",
                result_train["prec"],
                "; test_prec: ",
                result_test["prec"],
                "; train_recall: ",
                result_train["recall"],
                "; test_recall: ",
                result_test["recall"],
                "; train_acc: ",
                result_train["acc"],
                "; test_acc: ",
                result_test["acc"],
                "; epoch time: ",
                "{0:0.2f}".format(elapsed),
            )
        elif args.multi_label and epoch % 10 == 0:
            print(
                "epoch: ",
                epoch,
                "; loss: ",
                loss.item(),
                "; train_prec: ",
                result_train["prec"],
                "; test_prec: ",
                result_test["prec"],
                "; train_recall: ",
                result_train["recall"],
                "; test_recall: ",
                result_test["recall"],
                # "; train_F1: ",
                # result_train["F1"],
                # "; test_F1: ",
                # result_test["F1"],
                "; train_AUC: ",
                result_train["AUC"],
                "; test_AUC: ",
                result_test["AUC"],
                "; epoch time: ",
                "{0:0.2f}".format(elapsed),
            )

        if scheduler is not None:
            scheduler.step()

    # computation graph
    model.eval()
    if args.gpu:
        ypred_train = model(x.cuda(), adj.cuda(), train_data.cuda())
        ypred_test = model(x.cuda(), adj.cuda(), test_data.cuda())
    else:
        ypred_train = model(x, adj, train_data)
        ypred_test = model(x, adj, test_data)
    cg_data = {
        "graph": G,
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

    return model


def predict(original_graph, predict_data, model, args):
    # predict in original_graph instead of graph that has deleted edges in training data
    start_time = time.time()
    num_nodes = original_graph.number_of_nodes()
    adj_original_graph = nx.to_numpy_array(original_graph)
    adj_original_graph = adj_original_graph.transpose()

    existing_node = list(original_graph.nodes)[-1]
    feat_dim = original_graph.nodes[existing_node]["feat"].shape[0]
    x = np.zeros((num_nodes, feat_dim), dtype=float)
    for i, u in enumerate(original_graph.nodes()):
        x[i, :] = original_graph.nodes[u]["feat"]

    adj = np.expand_dims(adj_original_graph, axis=0)
    x = np.expand_dims(x, axis=0)

    adj = torch.tensor(adj, dtype=torch.float)
    x = torch.tensor(x, requires_grad=True, dtype=torch.float)
    end_time = time.time()
    print("prepare inputs for model for predicting time: ", (end_time - start_time))

    model.eval()

    print("predict threshold: " + str(args.predict_threshold))

    start = 0
    end = start + args.predict_batch_size
    predicted_links = []
    while start < predict_data.shape[0]:
        if args.gpu:
            # ypred = model(x.cuda(), adj.cuda(), predict_data[start:end, :].cuda())
            ypred = model(x.cuda(), adj.cuda(), torch.tensor(predict_data[start:end, :], dtype=torch.long).cuda())
        else:
            ypred = model(x, adj, predict_data[start:end, :])
        if args.single_edge_label:
            rows_idx = np.where(torch.sigmoid(ypred.cpu())[:, :, 1:2] > args.predict_threshold)[1]
            pre_etype = np.array([0] * rows_idx.shape[0])         # '0' means edge label
            predicted_links.append(np.column_stack((predict_data[start:end, :][rows_idx], pre_etype)))
        elif args.multi_class:
            rows_idx = np.where(torch.max(torch.sigmoid(ypred.cpu()), 2)[0] > args.predict_threshold)[1]
            pre_etype = torch.argmax(torch.sigmoid(ypred.cpu()), 2)[0].detach().numpy()[rows_idx]
            predicted_links.append(np.column_stack((predict_data[start:end, :][rows_idx], pre_etype)))
        elif args.multi_label:
            for row in range(ypred.cpu().shape[1]):
                for elabel in range(ypred.cpu().shape[2]):
                    if ypred.cpu()[0][row][elabel] > args.predict_threshold:
                        src_id = predict_data[start:end, :][row][0]
                        dst_id = predict_data[start:end, :][row][1]
                        if not original_graph.has_edge(src_id, dst_id, elabel):
                            predicted_links.append((src_id, dst_id, elabel))
        start = end
        end = start + args.predict_batch_size

    predicted_link_results = predicted_links[0]
    for i in range(1, len(predicted_links)):
        predicted_link_results = np.concatenate((predicted_link_results, predicted_links[i]), axis=0)
    path = "data/" + args.dataset + "/" + args.dataset + ".link"
    np.savetxt(path, predicted_link_results, delimiter="\t", fmt="%d")

    # predicted_links = []
    # if args.single_edge_label:
    #     ypred = torch.sigmoid(ypred)[0].detach().numpy()
    #     for row in range(ypred.shape[0]):
    #         if ypred[row][1] > args.predict_threshold:
    #             predicted_links.append((predict_data[row][0], predict_data[row][1], 0))  # '0' means edge label
    # elif args.multi_class:
    #     ypred = torch.sigmoid(ypred)[0].detach().numpy()
    #     for row in range(ypred.shape[0]):
    #         if max(ypred[row]) > args.predict_threshold:
    #             predicted_links.append((predict_data[row][0], predict_data[row][1], np.argmax(ypred[row])))
    # elif args.multi_label:
    #     ypred = ypred[0].detach().numpy()
    #     for row in range(ypred.shape[0]):
    #         for elabel in range(ypred.shape[1]):
    #             if ypred[row][elabel] > args.predict_threshold:
    #                 if not original_graph.has_edge(predict_data[row][0], predict_data[row][1], elabel):
    #                     predicted_links.append((predict_data[row][0], predict_data[row][1], elabel))
    # predicted_links = np.array(predicted_links)
    # path = "data/" + args.dataset + "/" + args.dataset + ".link"
    # np.savetxt(path, predicted_links, delimiter="\t", fmt="%d")

    return predicted_links


def predict_batch(original_graph, num_edge_labels, model, args):

    def check_predict(predict_data, ypred, args):
        predicted_links = []
        if args.single_edge_label:
            rows_idx = np.where(torch.sigmoid(ypred.cpu())[:, :, 1:2] > args.predict_threshold)[1]
            pre_etype = np.array([0] * rows_idx.shape[0])  # '0' means edge label
            predicted_links.append(np.column_stack((predict_data[rows_idx], pre_etype)))
        elif args.multi_class:
            rows_idx = np.where(torch.max(torch.sigmoid(ypred.cpu()), 2)[0] > args.predict_threshold)[1]
            pre_etype = torch.argmax(torch.sigmoid(ypred.cpu()), 2)[0].detach().numpy()[rows_idx]
            predicted_links.append(np.column_stack((predict_data[rows_idx], pre_etype)))
        elif args.multi_label:
            for row in range(ypred.cpu().shape[1]):
                for elabel in range(ypred.cpu().shape[2]):
                    if ypred.cpu()[0][row][elabel] > args.predict_threshold:
                        src_id = predict_data[row][0]
                        dst_id = predict_data[row][1]
                        if not original_graph.has_edge(src_id, dst_id, elabel):
                            predicted_links.append((src_id, dst_id, elabel))
        return predicted_links

    start_time = time.time()
    num_nodes = original_graph.number_of_nodes()
    adj_original_graph = nx.to_numpy_array(original_graph)
    adj_original_graph = adj_original_graph.transpose()

    existing_node = list(original_graph.nodes)[-1]
    feat_dim = original_graph.nodes[existing_node]["feat"].shape[0]
    x = np.zeros((num_nodes, feat_dim), dtype=float)
    for i, u in enumerate(original_graph.nodes()):
        x[i, :] = original_graph.nodes[u]["feat"]

    adj = np.expand_dims(adj_original_graph, axis=0)
    x = np.expand_dims(x, axis=0)

    adj = torch.tensor(adj, dtype=torch.float)
    x = torch.tensor(x, requires_grad=True, dtype=torch.float)
    end_time = time.time()
    print("prepare inputs for model for predicting time: ", (end_time - start_time))

    model.eval()
    print("predict threshold: " + str(args.predict_threshold))

    # generate data of a batch size, then make prediction
    num_predict_data = 0
    predict_data = []
    predicted_links = []
    if args.single_edge_label or args.multi_class:
        for src in range(num_nodes):
            for dst in range(num_nodes):
                if not original_graph.has_edge(src, dst):
                    predict_data.append((src, dst))

            if len(predict_data) >= args.predict_batch_size:
                num_predict_data = num_predict_data + len(predict_data)
                predict_data = np.array(predict_data)
                if args.gpu:
                    ypred = model(x.cuda(), adj.cuda(), torch.tensor(predict_data, dtype=torch.long).cuda())
                else:
                    ypred = model(x, adj, predict_data)
                predicted_links += check_predict(predict_data, ypred, args)
                predict_data = []

    elif args.multi_label:
        for src in range(num_nodes):
            for dst in range(num_nodes):
                # predict_data.append((src, dst))
                if not original_graph.has_edge(src, dst):
                    for elabel in range(num_edge_labels):
                        predict_data.append((src, dst))
                else:
                    for elabel in range(num_edge_labels):
                        if not original_graph.has_edge(src, dst, elabel):
                            predict_data.append((src, dst))
                            break

            if len(predict_data) >= args.predict_batch_size:
                num_predict_data = num_predict_data + len(predict_data)
                predict_data = np.array(predict_data)
                if args.gpu:
                    ypred = model(x.cuda(), adj.cuda(), torch.tensor(predict_data, dtype=torch.long).cuda())
                else:
                    ypred = model(x, adj, predict_data)
                predicted_links += check_predict(predict_data, ypred, args)
                predict_data = []

    if len(predict_data) > 0:
        num_predict_data = num_predict_data + len(predict_data)
        predict_data = np.array(predict_data)
        if args.gpu:
            ypred = model(x.cuda(), adj.cuda(), torch.tensor(predict_data, dtype=torch.long).cuda())
        else:
            ypred = model(x, adj, predict_data)
        predicted_links += check_predict(predict_data, ypred, args)

    print("generate ", num_predict_data, "predict data")
    predicted_link_results = predicted_links[0]
    for i in range(1, len(predicted_links)):
        predicted_link_results = np.concatenate((predicted_link_results, predicted_links[i]), axis=0)
    path = "data/" + args.dataset + "/" + args.dataset + ".link"
    np.savetxt(path, predicted_link_results, delimiter="\t", fmt="%d")


#############################
#
# Evaluate Trained Model
#
#############################
def evaluate_link(model, adj, x, test_data, test_labels, ypred_train, train_labels, args):
    test_labels = np.expand_dims(test_labels, axis=0)
    test_labels = torch.tensor(test_labels, dtype=torch.long)
    if args.gpu:
        ypred_test = model(x.cuda(), adj.cuda(), test_data.cuda())
    else:
        ypred_test = model(x, adj, test_data)

    if args.single_edge_label or args.multi_class:
        _, ypred_train_labels = torch.max(ypred_train, 2)
        ypred_train_labels = ypred_train_labels.numpy()

        _, ypred_test_labels = torch.max(ypred_test.cpu(), 2)
        ypred_test_labels = ypred_test_labels.numpy()

        pred_train = np.ravel(ypred_train_labels)
        pred_test = np.ravel(ypred_test_labels)
        labels_train = np.ravel(train_labels)
        labels_test = np.ravel(test_labels)

        result_train = {
            "prec": metrics.precision_score(labels_train, pred_train, average="macro"),
            "recall": metrics.recall_score(labels_train, pred_train, average="macro"),
            "acc": metrics.accuracy_score(labels_train, pred_train),
            "conf_mat": metrics.confusion_matrix(labels_train, pred_train),
        }
        result_test = {
            "prec": metrics.precision_score(labels_test, pred_test, average="macro"),
            "recall": metrics.recall_score(labels_test, pred_test, average="macro"),
            "acc": metrics.accuracy_score(labels_test, pred_test),
            "conf_mat": metrics.confusion_matrix(labels_test, pred_test),
        }
        return result_train, result_test

    elif args.multi_label:
        ypred_train = ypred_train.detach().numpy()[0]
        # ypred_train[ypred_train < 0.5] = 0
        # ypred_train[ypred_train >= 0.5] = 1
        # ypred_train = ypred_train.astype(int)

        ypred_test = ypred_test.cpu().detach().numpy()[0]
        # ypred_test[ypred_test < 0.5] = 0
        # ypred_test[ypred_test >= 0.5] = 1
        # ypred_test = ypred_test.astype(int)

        train_labels = train_labels.numpy()[0]
        test_labels = test_labels.numpy()[0]

        num_classes = ypred_train.shape[1]
        # For each class
        precision = dict()
        recall = dict()
        threshold = dict()
        average_precision = dict()
        for i in range(num_classes):
            precision[i], recall[i], threshold[i] = metrics.precision_recall_curve(train_labels[:, i], ypred_train[:, i])
            average_precision[i] = metrics.average_precision_score(train_labels[:, i], ypred_train[:, i])
        # A "micro-average": quantifying score on all classes jointly
        # precision["micro"], recall["micro"], _ = metrics.precision_recall_curve(train_labels.ravel(), ypred_train.ravel())
        # average_precision["micro"] = metrics.average_precision_score(train_labels, ypred_train, average="micro")
        # average_precision_train = average_precision["micro"]

        precision_sum = 0.0
        recall_sum = 0.0
        for i in range(num_classes):
            precision_sum += sum(precision[i][:(precision[i].shape[0] - 1)]) / (precision[i].shape[0] - 1)
            recall_sum += sum(recall[i][:(recall[i].shape[0] - 1)]) / (recall[i].shape[0] - 1)
        prec_train = precision_sum / num_classes
        rec_train = recall_sum / num_classes
        F1_train = 2 * prec_train * rec_train / (prec_train + rec_train)

        # For each class
        precision = dict()
        recall = dict()
        threshold = dict()
        average_precision = dict()
        for i in range(num_classes):
            precision[i], recall[i], threshold[i] = metrics.precision_recall_curve(test_labels[:, i], ypred_test[:, i])
            average_precision[i] = metrics.average_precision_score(test_labels[:, i], ypred_test[:, i])
        # # A "micro-average": quantifying score on all classes jointly
        # precision["micro"], recall["micro"], _ = metrics.precision_recall_curve(test_labels.ravel(), ypred_test.ravel())
        # average_precision["micro"] = metrics.average_precision_score(test_labels, ypred_test, average="micro")
        # average_precision_test = average_precision["micro"]

        precision_sum = 0.0
        recall_sum = 0.0
        for i in range(num_classes):
            precision_sum += sum(precision[i][:(precision[i].shape[0] - 1)]) / (precision[i].shape[0] - 1)
            recall_sum += sum(recall[i][:(recall[i].shape[0] - 1)]) / (recall[i].shape[0] - 1)
        prec_test = precision_sum / num_classes
        rec_test = recall_sum / num_classes
        F1_test = 2 * prec_test * rec_test / (prec_test + rec_test)

        # auc_train = metrics.roc_curve(train_labels.ravel(), ypred_train.ravel())
        # auc_test = metrics.roc_curve(test_labels.ravel(), ypred_test.ravel())
        if (train_labels.shape[-1] != 1):
            auc_train = metrics.roc_auc_score(train_labels, ypred_train)
            auc_test = metrics.roc_auc_score(test_labels, ypred_test)
        else:
            auc_train = 0
            auc_test = 0

        result_train = {
            "prec": prec_train,
            "recall": rec_train,
            "F1": F1_train,
            "AUC": auc_train,
        }
        result_test = {
            "prec": prec_test,
            "recall": rec_test,
            "F1": F1_test,
            "AUC": auc_test,
        }
        return result_train, result_test


def link_prediction_task(args, writer=None):
    start_time = time.time()
    graph, node_labels, num_node_labels, num_edge_labels = io_utils.read_graphfile(
        args.datadir, args.dataset, args
    )
    end_time = time.time()
    print("load graph time: ", (end_time - start_time))
    input_dim = graph.graph["feat_dim"]

    original_graph = copy.deepcopy(graph)

    start_time = time.time()
    graph, train_data, train_labels, test_data, test_labels = prepare_data(graph, num_edge_labels, args)
    end_time = time.time()
    print("prepare training and testing data time: ", (end_time - start_time))

    start_time = time.time()
    model = None
    if args.model == "GcnEncoderNode":
        if args.single_edge_label:
            model = models.GcnEncoderNode(
                input_dim,
                args.hidden_dim,
                args.output_dim,
                2,   # binary classification for link prediction.
                args.num_gc_layers,
                bn=args.bn,
                dropout=args.dropout,
                args=args,
            )
        elif args.multi_label or args.multi_class:
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
    elif args.model == "ogb_GCN":
        model = ogb_models.GCN(
            input_dim,
            args.hidden_dim,
            args.output_dim,
            num_layers=args.num_gc_layers,
            dropout=args.dropout,
            feature_dim=2 if args.single_edge_label else num_edge_labels,
            args=args,
        )

    if args.gpu:
        model = model.cuda()
    end_time = time.time()
    print("construct model time: ", (end_time - start_time))

    start_time = time.time()
    model = train_link_classifier(graph, node_labels, train_data, train_labels, test_data, test_labels, model, args, writer=writer)
    end_time = time.time()
    print("whole training and testing time: ", (end_time - start_time))

    # method-1: generate all predicted data, then make prediction in a batch way
    # start_time = time.time()
    # predict_data = generate_predict_data(original_graph, num_edge_labels, args)
    # end_time = time.time()
    # print("generate predict data time: ", (end_time - start_time))

    # start_time = time.time()
    # predict(original_graph, predict_data, model, args)
    # end_time = time.time()
    # print("whole predicting time: ", (end_time - start_time))

    # method-2: generate data of a batch size, then make prediction
    start_time = time.time()
    #predict_batch(original_graph, num_edge_labels, model, args)
    end_time = time.time()
    print("whole predicting time: ", (end_time - start_time))


def main():
    start_time = time.time()
    prog_args = configs.arg_parse()

    path = os.path.join(prog_args.logdir, io_utils.gen_prefix(prog_args))
    # writer = SummaryWriter(path)

    if prog_args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = prog_args.cuda
        print("CUDA", prog_args.cuda)
    else:
        print("Using CPU")

    if prog_args.link_prediction is True:
        link_prediction_task(prog_args, writer=None)
        # link_prediction_task(prog_args, writer=writer)

    # writer.close()

    end_time = time.time()
    print("Total running time: ", (end_time - start_time))


if __name__ == "__main__":
    setup_seed(1226)
    main()

