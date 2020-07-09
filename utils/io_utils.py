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
        graph, index, src_dst_explanation, link_label,
        edge_threshold=None, feat_threshold=None, edge_num_threshold=None,
        args=None
):

    src_idx = 0
    dst_idx = 1
    adj = src_dst_explanation["pattern_adj"]
    feat = src_dst_explanation["pattern_feat"]
    neighbors = src_dst_explanation["pattern_nodes"]

    if len(adj[adj > edge_threshold]) == 0:
        return None

    # threshold definition-1
    threshold = threshold_num = 0
    if edge_num_threshold is not None:
        neigh_size = len(adj[adj > 0])
        if neigh_size == 0:
            return None
        threshold_num = min(neigh_size, edge_num_threshold)
        threshold = np.sort(adj[adj > 0])[-threshold_num]   # the threshold_num - th largest value in adj
    if edge_threshold < threshold:
        edge_threshold = threshold

    # threshold definition-2: use average threshold
    avg_threshold = sum(adj[adj > 0]) / adj[adj > 0].shape[0]
    if edge_threshold < avg_threshold:
        edge_threshold = avg_threshold

    # To get the max connected subgraph, thus use undirected graph.
    pattern = nx.Graph()

    reserved_edge_list = []
    reserved_node_list = []
    adj_sorted_values = -np.sort(-adj[adj > 0])
    flag = 0
    for i in range(adj_sorted_values.shape[0]):
        # if len(reserved_edge_list) >= threshold_num:
        #     break
        # if adj_sorted_values[i] < edge_threshold and src_idx in reserved_node_list and dst_idx in reserved_node_list and nx.is_connected(pattern):
        #     break
        position = np.where(adj == adj_sorted_values[i])
        for j in range(position[0].shape[0]):
            src = position[1][j]
            dst = position[0][j]
            reserved_node_list.append(src)
            reserved_node_list.append(dst)
            reserved_edge_list.append((src, dst))
            pattern.add_edges_from([(src, dst)])

            if len(reserved_edge_list) >= threshold_num:
                flag = 1
                break

            if src_idx in reserved_node_list and dst_idx in reserved_node_list and nx.is_connected(pattern):
                flag = 1
                break
        if flag == 1:
            break

    # largest_cc = max(nx.connected_components(pattern), key=len)
    if not nx.is_connected(pattern) or len(pattern) == 0:
        return None

    reserved_nodes = np.unique(reserved_node_list)
    reserved_feat = feat[reserved_nodes]

    if src_idx not in reserved_nodes and dst_idx not in reserved_nodes:
        return None

    reserved_feat = torch.sigmoid(torch.tensor(reserved_feat)).detach().numpy()
    reserved_feat[reserved_feat >= feat_threshold] = 1
    reserved_feat[reserved_feat < feat_threshold] = 0

    # renumber.
    map_nodes = {}
    for i in range(reserved_nodes.shape[0]):
        map_nodes[reserved_nodes[i]] = i
        reserved_nodes[i] = i
    reserved_edges = []
    for edge in reserved_edge_list:
        src_oid = neighbors[edge[0]]
        dst_oid = neighbors[edge[1]]
        src_new_idx = map_nodes[edge[0]]
        dst_new_idx = map_nodes[edge[1]]
        if len(graph[src_oid][dst_oid]) == 1:
            reserved_edges.append((src_new_idx, dst_new_idx, graph[src_oid][dst_oid][0]["label"]))
        else:
            for j in reversed(range(len(graph[src_oid][dst_oid]))):
                reserved_edges.append((src_new_idx, dst_new_idx, graph[src_oid][dst_oid][j]["label"]))

    denoise_result = {
        "reserved_nodes": reserved_nodes,
        "reserved_edge_list": reserved_edges,
        "reserved_feat": reserved_feat,
    }

    # 3. write the explanation results into file.
    path = "explanations/"
    src_label = graph.nodes[neighbors[src_idx]]["label"]
    dst_label = graph.nodes[neighbors[dst_idx]]["label"]
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
        with open(path + args.dataset + ".explanation_" + suffix + "_" + str(args.n_hops) + "hops", "a+") as f:
            f.write("#\t" + str(index) + "\n")
            for node_oid, node_id in map_nodes.items():
                f.write("v\t" + str(node_id) + "\t" + str(graph.nodes()[neighbors[node_oid]]["label"]))
                for attr_id in range(reserved_feat[node_id].shape[0]):
                    if reserved_feat[node_id][attr_id] == 1:
                        f.write("\t" + str(attr_id))
                f.write("\n")
            for edge in reserved_edges:
                f.write("e\t" + str(edge[0]) + "\t" + str(edge[1]) + "\t" + str(edge[2]) + "\n")
        f.close()

    return denoise_result


def combine_src_dst_explanations(
        src_explanation_results, dst_explanation_results
):
    src_masked_feat = src_explanation_results["src_masked_feat"]
    src_masked_adj = src_explanation_results["src_masked_adj"]
    src_idx_new = src_explanation_results["src_idx_new"]
    src_neighbors = src_explanation_results["src_neighbors"]

    dst_masked_feat = dst_explanation_results["dst_masked_feat"]
    dst_masked_adj = dst_explanation_results["dst_masked_adj"]
    dst_idx_new = dst_explanation_results["dst_idx_new"]
    dst_neighbors = dst_explanation_results["dst_neighbors"]

    pattern_nodes = np.concatenate((src_neighbors, dst_neighbors), axis=0).tolist()
    pattern_nodes = np.unique(pattern_nodes)
    num_nodes = pattern_nodes.shape[0]
    num_src_nodes = src_masked_adj.shape[0]
    num_dst_nodes = dst_masked_adj.shape[0]

    oid2newid_src = {}
    oid2newid_dst = {}
    for i in range(src_neighbors.shape[0]):
        oid2newid_src[src_neighbors[i]] = i
    for j in range(dst_neighbors.shape[0]):
        oid2newid_dst[dst_neighbors[j]] = j

    # 2. renumber.
    pattern_nodes = filter(lambda x: x != src_neighbors[src_idx_new] and x != dst_neighbors[dst_idx_new], pattern_nodes)
    pattern_nodes = [i for i in pattern_nodes]
    pattern_nodes = np.unique(pattern_nodes)
    map_nodes = {src_neighbors[src_idx_new]: 0, dst_neighbors[dst_idx_new]: 1}
    for i in range(pattern_nodes.shape[0]):
        map_nodes[pattern_nodes[i]] = i + 2
    pattern_nodes = np.insert(pattern_nodes, 0, dst_neighbors[dst_idx_new])
    pattern_nodes = np.insert(pattern_nodes, 0, src_neighbors[src_idx_new])

    # adj_list = []
    # for i in range(num_src_nodes):
    #     for j in range(num_src_nodes):
    #         if src_masked_adj[i, j] > 0:
    #             adj_list.append((map_nodes[src_neighbors[j]], map_nodes[src_neighbors[i]]))
    # for i in range(num_dst_nodes):
    #     for j in range(num_dst_nodes):
    #         if dst_masked_adj[i, j] > 0:
    #             adj_list.append((map_nodes[dst_neighbors[j]], map_nodes[dst_neighbors[i]]))
    #
    # pattern = nx.MultiDiGraph()
    # pattern.add_nodes_from(range(num_nodes))
    # pattern.add_edges_from(adj_list)
    # pattern_adj = nx.to_numpy_array(pattern)
    # pattern_adj = pattern_adj.transpose()

    pattern_adj = np.zeros((num_nodes, num_nodes))
    for i in range(num_src_nodes):
        for j in range(num_src_nodes):
            if src_masked_adj[i, j] > 0:
                pattern_adj[map_nodes[src_neighbors[i]]][map_nodes[src_neighbors[j]]] = src_masked_adj[i, j]
    for i in range(num_dst_nodes):
        for j in range(num_dst_nodes):
            if dst_masked_adj[i, j] > 0:
                weight = pattern_adj[map_nodes[dst_neighbors[i]]][map_nodes[dst_neighbors[j]]]
                if weight < dst_masked_adj[i, j]:
                    pattern_adj[map_nodes[dst_neighbors[i]]][map_nodes[dst_neighbors[j]]] = dst_masked_adj[i, j]

    src_idx_new = 0
    dst_idx_new = 1

    pattern_feat = np.zeros((num_nodes, src_masked_feat.shape[1]))
    for pattern_v in pattern_nodes:
        if pattern_v in oid2newid_src.keys():
            pattern_feat[map_nodes[pattern_v]] = src_masked_feat[oid2newid_src[pattern_v]]
        if pattern_v in oid2newid_dst.keys():
            if pattern_feat[map_nodes[pattern_v]][0] != 0:
                for feat_dim in range(src_masked_feat.shape[1]):
                    if pattern_feat[map_nodes[pattern_v]][feat_dim] < dst_masked_feat[oid2newid_dst[pattern_v]][feat_dim]:
                        pattern_feat[map_nodes[pattern_v]][feat_dim] = dst_masked_feat[oid2newid_dst[pattern_v]][feat_dim]
            else:
                pattern_feat[map_nodes[pattern_v]] = dst_masked_feat[oid2newid_dst[pattern_v]]

    pattern = {
        "pattern_nodes": pattern_nodes,
        "pattern_adj": pattern_adj,
        "pattern_feat": pattern_feat,
    }
    return pattern


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
