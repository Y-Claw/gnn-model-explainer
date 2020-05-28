""" io_utils.py

    Utilities for reading and writing logs.
"""
import os
import statistics
import re
import csv

import numpy as np
import pandas as pd
import scipy as sc


import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import torch
import networkx as nx
import tensorboardX

import cv2

import torch
import torch.nn as nn
from torch.autograd import Variable
import random

# Only necessary to rebuild the Chemistry example
# from rdkit import Chem

import utils.featgen as featgen

use_cuda = torch.cuda.is_available()


def gen_prefix(args):
    '''Generate label prefix for a graph model.
    '''
    if args.bmname is not None:
        name = args.bmname
    else:
        name = args.dataset
    name += "_" + args.method

    name += "_h" + str(args.hidden_dim) + "_o" + str(args.output_dim)
    if not args.bias:
        name += "_nobias"
    if len(args.name_suffix) > 0:
        name += "_" + args.name_suffix
    return name


def gen_explainer_prefix(args):
    '''Generate label prefix for a graph explainer model.
    '''
    name = gen_prefix(args) + "_explain"
    if len(args.explainer_suffix) > 0:
        name += "_" + args.explainer_suffix
    return name


def create_filename(save_dir, args, isbest=False, num_epochs=-1):
    """
    Args:
        args        :  the arguments parsed in the parser
        isbest      :  whether the saved model is the best-performing one
        num_epochs  :  epoch number of the model (when isbest=False)
    """
    filename = os.path.join(save_dir, gen_prefix(args))
    os.makedirs(filename, exist_ok=True)

    if isbest:
        filename = os.path.join(filename, "best")
    elif num_epochs > 0:
        filename = os.path.join(filename, str(num_epochs))

    return filename + ".pth.tar"


def save_checkpoint(model, optimizer, args, num_epochs=-1, isbest=False, cg_dict=None):
    """Save pytorch model checkpoint.
    
    Args:
        - model         : The PyTorch model to save.
        - optimizer     : The optimizer used to train the model.
        - args          : A dict of meta-data about the model.
        - num_epochs    : Number of training epochs.
        - isbest        : True if the model has the highest accuracy so far.
        - cg_dict       : A dictionary of the sampled computation graphs.
    """
    filename = create_filename(args.ckptdir, args, isbest, num_epochs=num_epochs)
    torch.save(
        {
            "epoch": num_epochs,
            "model_type": args.method,
            "optimizer": optimizer,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "cg": cg_dict,
        },
        filename,
    )


def load_ckpt(args, isbest=False):
    '''Load a pre-trained pytorch model from checkpoint.
    '''
    print("loading model")
    filename = create_filename(args.ckptdir, args, isbest)
    print(filename)
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        ckpt = torch.load(filename)
    else:
        print("Checkpoint does not exist!")
        print("Checked path -- {}".format(filename))
        print("Make sure you have provided the correct path!")
        print("You may have forgotten to train a model for this dataset.")
        print()
        print("To train one of the paper's models, run the following")
        print(">> python train.py --dataset=DATASET_NAME")
        print()
        raise Exception("File not found.")
    return ckpt


def denoise_adj_feat(
        graph, adj, node_idx, feat, neighbors,
        edge_threshold=None, feat_threshold=None, edge_num_threshold=None,
        args=None
):
    """Cleaning a graph by thresholding its node values.

    Args:
        - adj               :  Adjacency matrix.
        - node_idx          :  Index of node to highlight (TODO ?)
        - feat              :  An array of node features.
        - label             :  A list of node labels.
        - threshold         :  The weight threshold.
        - max_component     :  TODO
    """

    num_nodes = adj.shape[0]

    # threshold definition-1
    threshold = 0
    if edge_num_threshold is not None:
        # this is for symmetric graphs: edges are repeated twice in adj
        adj_threshold_num = edge_num_threshold * 2  # undirected graph
        if args.directed_graph is True:
            adj_threshold_num = edge_num_threshold
        # adj += np.random.rand(adj.shape[0], adj.shape[1]) * 1e-4
        neigh_size = len(adj[adj > 0])
        if neigh_size == 0:
            return None
        threshold_num = min(neigh_size, adj_threshold_num)
        threshold = np.sort(adj[adj > 0])[-threshold_num]   # the threshold_num - th largest value in adj
    edge_threshold = edge_threshold if edge_threshold > threshold else threshold

    # threshold definition-2: use average threshold
    avg_threshold = sum(adj[adj > 0]) / adj[adj > 0].shape[0]
    edge_threshold = threshold if threshold > avg_threshold else avg_threshold
    # edge_threshold = edge_threshold if edge_threshold > 0.5 else 0.5

    reserved_edge_list = []
    reserved_node_list = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if adj[i, j] < edge_threshold:
                continue
            src_idx = neighbors[j]
            dst_idx = neighbors[i]
            reserved_node_list.append(i)
            reserved_node_list.append(j)
            if len(graph[src_idx][dst_idx]) == 1:
                reserved_edge_list.append((src_idx, dst_idx, graph[src_idx][dst_idx][0]["label"]))
            else:
                for idx in reversed(range(len(graph[src_idx][dst_idx]))):
                    reserved_edge_list.append(graph[src_idx][dst_idx][idx]['label'])
    if len(reserved_node_list) == 0:
        return None
    reserved_nodes = np.unique(reserved_node_list)
    reserved_feat = feat[reserved_nodes]

    reserved_nodes = neighbors[reserved_nodes]   # idx -> original id in graph
    if neighbors[node_idx] not in reserved_nodes:
        return None
    node_idx_new = np.where(reserved_nodes == neighbors[node_idx])[0][0]

    feat_threshold = 0.5
    reserved_feat = torch.sigmoid(torch.tensor(reserved_feat)).detach().numpy()
    reserved_feat[reserved_feat >= feat_threshold] = 1
    reserved_feat[reserved_feat < feat_threshold] = 0

    denoise_result = {
        "reserved_nodes": reserved_nodes,
        "reserved_edge_list": reserved_edge_list,
        "reserved_feat": reserved_feat,
        "node_idx_new": node_idx_new,
    }

    return denoise_result


def combine_src_dst_explanations(
        graph, index, src_idx, dst_idx, link_label, src_denoise_result, dst_denoise_result, args
):
    # 1. combine explanations for src and dst in link(src, dst)
    pattern_nodes = np.concatenate((src_denoise_result["reserved_nodes"], dst_denoise_result["reserved_nodes"]), axis=0)
    pattern_edges = src_denoise_result["reserved_edge_list"] + dst_denoise_result["reserved_edge_list"]
    src_features = src_denoise_result["reserved_feat"]
    dst_features = dst_denoise_result["reserved_feat"]
    pattern_edges = np.unique(pattern_edges, axis=0)

    oid2newid_src = {}
    oid2newid_dst = {}
    for i in range(src_denoise_result["reserved_nodes"].shape[0]):
        oid2newid_src[src_denoise_result["reserved_nodes"][i]] = i
    for j in range(dst_denoise_result["reserved_nodes"].shape[0]):
        oid2newid_dst[dst_denoise_result["reserved_nodes"][j]] = j

    # 2. renumber.
    pattern_nodes = filter(lambda x: x != src_idx and x != dst_idx, pattern_nodes)
    pattern_nodes = [i for i in pattern_nodes]
    pattern_nodes = np.unique(pattern_nodes)
    map_nodes = {src_idx: 0, dst_idx: 1}
    for i in range(pattern_nodes.shape[0]):
        map_nodes[pattern_nodes[i]] = i + 2

    # 3. write the explanation results into file.
    path = "explanations/"
    src_label = graph.nodes()[src_idx]["label"]
    dst_label = graph.nodes()[dst_idx]["label"]
    suffix = str(src_label) + "_" + str(dst_label)
    link_type_set = []  # link type
    if args.single_edge_label:
        link_type_set.append(0)
    elif args.multi_class:
        link_type_set.append(link_label)
    elif args.multi_label:
        non_zero_idx = np.where(link_label == 1)[0]  # one-hot
        for tmp_index in non_zero_idx:
            link_type_set.append(tmp_index)

    for link_type in link_type_set:
        suffix = suffix + "_" + str(link_type)
        with open(path + args.dataset + ".explaination_" + suffix + "_" + str(args.n_hops) + "hops", "a+") as f:
            f.write("#\t" + str(index) + "\n")
            for node_oid, node_id in map_nodes.items():
                f.write("v\t" + str(node_id) + "\t" + str(graph.nodes()[node_oid]["label"]))
                important_attr = []
                if node_oid in oid2newid_src.keys():
                    node_oid_feat = src_features[oid2newid_src[node_oid]]
                    for i in range(node_oid_feat.shape[0]):
                        if node_oid_feat[i] == 1:
                            important_attr.append(i)
                if node_oid in oid2newid_dst.keys():
                    node_oid_feat = dst_features[oid2newid_dst[node_oid]]
                    for i in range(node_oid_feat.shape[0]):
                        if node_oid_feat[i] == 1:
                            important_attr.append(i)
                important_attr = np.unique(important_attr)
                for attr_id in important_attr:
                    f.write("\t" + str(attr_id))
                f.write("\n")
            # for node_id in range(len(map_nodes)):
            #     f.write("v\t" + str(node_id) + "\t" + str(self.graph.nodes()[node_id]["label"]) + "\n")
            for edge in pattern_edges:
                f.write("e\t" + str(map_nodes[edge[0]]) + "\t" + str(map_nodes[edge[1]]) + "\t" + str(edge[2]) + "\n")
        f.close()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def is_number_str(numStr):
    flag = False
    numStr = str(numStr).strip().lstrip('-').lstrip('+')
    if str(numStr).isdigit():
        flag = True
    return flag


def is_float_str(numStr):
    flag = False
    numStr = str(numStr).strip().lstrip('-').lstrip('+')
    try:
        reg = re.compile(r'^[-+]?[0-9]+\.[0-9]+$')
        res = reg.match(str(numStr))
        if res:
            flag = True
    except Exception as ex:
        print("is_float() - error: " + str(ex))
    return flag


def attribute2vec(attrs):
    tmp_list = []
    for i in range(len(attrs)):
        if attrs[i] == "" or attrs[i] == "-":
            tmp_list.append(0)
        else:
            # filtered_attr = re.sub("[-_()\\\\,`\'\[\]]", "", attrs[i])
            filtered_attr = attrs[i]
            if is_number_str(filtered_attr):
                tmp_list.append(int(filtered_attr))
            elif is_float_str(filtered_attr):
                tmp_list.append(float(filtered_attr))
            else:
                tmp_list.append(len(filtered_attr) + random.random())
    return np.array(tmp_list)


def read_graphfile(datadir, dataname, args):
    """ Read data from https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets
        graph index starts with 1 in file

    Returns:
        List of networkx objects with graph and node labels
    """
    prefix = os.path.join(datadir, dataname, dataname)

    # filename_nodes = prefix + "_node_labels.txt"
    # node_labels = []
    # min_label_val = None
    # try:
    #     with open(filename_nodes) as f:
    #         for line in f:
    #             line = line.strip("\n")
    #             l = int(line)
    #             node_labels += [l]
    #             if min_label_val is None or min_label_val > l:
    #                 min_label_val = l
    #     # assume that node labels are consecutive
    #     num_unique_node_labels = max(node_labels) - min_label_val + 1
    #     node_labels = [l - min_label_val for l in node_labels]
    # except IOError:
    #     print("No node labels")

    # filename_node_attrs = prefix + "_node_attributes.txt"
    # node_attrs = []
    # try:
    #     with open(filename_node_attrs) as f:
    #         for line in f:
    #             line = line.strip("\s\n")
    #             attrs = [
    #                 float(attr) for attr in re.split("[,\s]+", line) if not attr == ""
    #             ]
    #             node_attrs.append(np.array(attrs))
    # except IOError:
    #     print("No node attributes")

    # if edge_labels:
    #     # For Tox21_AHR we want to know edge labels
    #     filename_edges = prefix + "_edge_labels.txt"
    #     edge_labels = []
    #
    #     edge_label_vals = []
    #     with open(filename_edges) as f:
    #         for line in f:
    #             line = line.strip("\n")
    #             val = int(line)
    #             if val not in edge_label_vals:
    #                 edge_label_vals.append(val)
    #             edge_labels.append(val)
    #
    #     edge_label_map_to_int = {val: i for i, val in enumerate(edge_label_vals)}

    # filename_adj = prefix + "_A.txt"
    # adj_list = []
    # # edge_label_list={i:[] for i in range(1,len(graph_labels)+1)}
    # num_edges = 0
    # with open(filename_adj) as f:
    #     for line in f:
    #         line = line.strip("\n").split(",")
    #         src, dst = (int(line[0].strip(" ")), int(line[1].strip(" ")))
    #         adj_list.append((src, dst))
    #         # edge_label_list[graph_indic[e0]].append(edge_labels[num_edges])
    #         num_edges += 1

    filename_v = prefix + ".v"  # id, label, attr1, attr2, ...
    node_ids = []
    node_labels = []
    node_attrs = []
    with open(filename_v) as f:
        for line in f:
            if line == "\n":
                continue
            line = line.strip("\n").split("\t")
            node_ids.append(int(line[0]))
            node_labels.append(int(line[1]))
            # node_attrs.append(attribute2vec(line[2:]))
            attr_list = []
            for attr in line[2:]:
                attr = attr.split(" ")  # for k-dimension feature vectors (k > 1)
                if len(attr) == 1:
                    attr_list.append(float(attr[0]))
                else:
                    for element in attr:
                        attr_list.append(float(element))
            node_attrs.append(np.array(attr_list))
    num_node_labels = max(node_labels) + 1

    filename_e = prefix + ".e"  # (src_label, dst_label, edge_label)
    adj_list = []
    edge_labels = []
    num_edges = 0
    with open(filename_e) as f:
        for line in f:
            line = line.strip("\n").split("\t")
            src, dst, elabel = int(line[0]), int(line[1]), int(line[2])
            adj_list.append((src, dst, dict(label=elabel)))
            edge_labels.append(elabel)
            num_edges += 1
    # the order of edge_labels for graph.edges() is not corresponding!
    num_edge_labels = max(edge_labels) + 1

    # directed graph
    G = nx.MultiDiGraph()
    G.add_nodes_from(node_ids)
    G.add_edges_from(adj_list)

    # if not args.multi_label:
    #     for u in G.nodes():
    #         G.nodes[u]["label"] = node_labels[u]
    #         G.nodes[u]["feat"] = node_attrs[u]
    # else:
    #     for u in G.nodes():
    #         if num_node_labels > 0:
    #             node_label_one_hot = [0] * num_node_labels
    #             node_label = node_labels[u]
    #             node_label_one_hot[node_label] = 1
    #             G.nodes[u]["label"] = node_label_one_hot
    #         if len(node_attrs) > 0:
    #             G.nodes[u]["feat"] = node_attrs[u]
    for u in G.nodes():
        G.nodes[u]["label"] = node_labels[u]
        G.nodes[u]["feat"] = node_attrs[u]
    if len(node_attrs) > 0:
        G.graph["feat_dim"] = node_attrs[0].shape[0]

    # # relabeling
    # mapping = {}
    # it = 0
    # if float(nx.__version__) < 2.0:
    #     for n in G.nodes():
    #         mapping[n] = it
    #         it += 1
    # else:
    #     for n in G.nodes:
    #         mapping[n] = it
    #         it += 1
    #
    # # indexed from 0
    # G = nx.relabel_nodes(G, mapping)

    return G, node_labels, num_node_labels, num_edge_labels
