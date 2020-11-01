import torch
import networkx as nx
import numpy as np
import multiprocessing as mp
import random
import sklearn
import torch.nn as nn
import os


def precompute_dist_data(edge_index, edge_weight, num_nodes,approximate=0):
    '''
    Here dist is 1/real_dist, higher actually means closer, 0 means disconnected
    :return:
    '''
    graph = nx.Graph()
    edge_list = edge_index.transpose(1, 0).tolist()
    for i in range(edge_weight.shape[0]):
        edge_list[i].append(edge_weight[i])
    graph.add_weighted_edges_from(edge_list)

    n = num_nodes
    dists_array = torch.zeros((n, n))
    # dists_dict = nx.all_pairs_shortest_path_length(graph,cutoff=approximate if approximate>0 else None)
    # dists_dict = {c[0]: c[1] for c in dists_dict}
    dists_dict = all_pairs_shortest_path_length_parallel(graph, cutoff=approximate if approximate > 0 else None)
    for i, node_i in enumerate(graph.nodes()):
        shortest_dist = dists_dict[node_i]
        for j, node_j in enumerate(graph.nodes()):
            dist = shortest_dist.get(node_j, -1)
            if dist != -1:
                # dists_array[i, j] = 1 / (dist + 1)
                dists_array[node_i, node_j] = 1 / (dist + 1)
    return dists_array

def single_source_shortest_path_length_range(graph, node_range, cutoff):
    dists_dict = {}
    for node in node_range:
        dists_dict[node] = nx.single_source_dijkstra(graph, node, cutoff)[0]
    return dists_dict

def merge_dicts(dicts):
    result = {}
    for dictionary in dicts:
        result.update(dictionary)
    return result

def all_pairs_shortest_path_length_parallel(graph,cutoff=None,num_workers=4):
    nodes = list(graph.nodes)
    dists_dict = single_source_shortest_path_length_range(graph, nodes, cutoff)
    return dists_dict

def get_random_anchorset(n,c=0.5, set_n = None):
    m = int(np.log2(n)) if set_n == None else int(np.log2(set_n))
    copy = int(c*m)
    anchorset_id = []
    for i in range(m):
        anchor_size = int((n if set_n == None else set_n)/np.exp2(i + 1))
        for j in range(copy):
            anchorset_id.append(np.random.choice(n,size=min(n, anchor_size),replace=False))
    return anchorset_id

def get_dist_max(anchorset_id, dist, device):
    dist_max = torch.zeros((dist.shape[0],len(anchorset_id))).to(device)
    dist_argmax = torch.zeros((dist.shape[0],len(anchorset_id))).long().to(device)
    for i in range(len(anchorset_id)):
        temp_id = anchorset_id[i]
        dist_temp = dist[:, temp_id]
        dist_max_temp, dist_argmax_temp = torch.max(dist_temp, dim=-1)
        dist_max[:,i] = dist_max_temp
        dist_argmax[:,i] = dist_argmax_temp
    return dist_max, dist_argmax

def preselect_anchor(data, layer_num=1, anchor_num=32, anchor_size_num=4, device='cpu', set_num=None):

    data.anchor_size_num = anchor_size_num
    data.anchor_set = []
    anchor_num_per_size = anchor_num//anchor_size_num
    for i in range(anchor_size_num):
        anchor_size = 2**(i+1)-1
        anchors = np.random.choice(data.num_nodes, size=(layer_num,anchor_num_per_size,anchor_size), replace=True)
        data.anchor_set.append(anchors)
    data.anchor_set_indicator = np.zeros((layer_num, anchor_num, data.num_nodes), dtype=int)

    anchorset_id = get_random_anchorset(data.num_nodes,c=1, set_n=set_num)
    data.dists_max, data.dists_argmax = get_dist_max(anchorset_id, data.dists, device)