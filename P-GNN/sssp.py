import random
import time

import networkx as nx
import multiprocessing as mp
import numpy as np
import os
import scipy.sparse as sp
import torch


def single_source_shortest_path_length_range(graph, node_range, cutoff):
    dists_dict = {}
    for node in node_range:
        start = time.time()
        dists_dict[node] = nx.single_source_shortest_path_length(graph, node, cutoff)
        end = time.time()
        print('finish in ', end - start)
    return dists_dict

def merge_dicts(dicts):
    result = {}
    for dictionary in dicts:
        result.update(dictionary)
    return result

def all_pairs_shortest_path_length_parallel(graph,cutoff=None,num_workers=4):
    nodes = list(graph.nodes)
    random.shuffle(nodes)
    if len(nodes)<50:
        num_workers = int(num_workers/4)
    elif len(nodes)<400:
        num_workers = int(num_workers/2)
    elif len(nodes) > 10000:
        num_workers = int(num_workers * 2)

    pool = mp.Pool(processes=num_workers)
    results = [pool.apply_async(single_source_shortest_path_length_range,
            args=(graph, nodes[int(len(nodes)/num_workers*i):int(len(nodes)/num_workers*(i+1))], cutoff)) for i in range(num_workers)]
    output = [p.get() for p in results]
    dists_dict = merge_dicts(output)
    pool.close()
    pool.join()
    return dists_dict

def precompute_dist_data(graph, num_nodes, approximate=0):
    '''
    Here dist is 1/real_dist, higher actually means closer, 0 means disconnected
    :return:
    '''

    n = num_nodes
    dists_array = sp.lil_matrix((n, n))
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

def explainer_load_graphs(name):
    prefix = os.path.join('..','data',name, name)
    filename_v = prefix + ".v"  # id, label, attr1, attr2, ...
    node_ids = []
    node_labels = []
    node_attrs = []
    num_node = 0
    with open(filename_v) as f:
        for line in f:
            if line == "\n":
                continue
            num_node += 1
            line = line.strip("\n").split("\t")
            node_ids.append(int(line[0]))
            node_labels.append(int(line[1]))
            # node_attrs.append(attribute2vec(line[2:]))
            attr_list = []
            for attr in line[1:]:#take node label as node attr
            #for attr in line[2:]:
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
    print('read map')
    label = sp.lil_matrix((num_node, num_node))
    G = nx.Graph()
    with open(filename_e) as f:
        for line in f:
            line = line.strip("\n").split("\t")
            src, dst, elabel = int(line[0]), int(line[1]), int(line[2])
            adj_list.append((src, dst))
            edge_labels.append(elabel)
            label[src,dst] = elabel + 1
            num_edges += 1
    # the order of edge_labels for graph.edges() is not corresponding!
    num_edge_labels = max(edge_labels)

    # directed graph
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
    graphs = []
    features = []
    edge_labels = []
    graphs.append(G)
    features.append(np.array(node_attrs))
    if num_edge_labels > 1:
        edge_labels.append(label)
    else:
        edge_labels.append(None)
    print('finish read map')
    return graphs, features, edge_labels


def load_tg_dataset(name='communities', type='PGNN'):
    graphs, features, edge_labels = explainer_load_graphs(name)
    return graphs


if __name__ == '__main__':
    #print(len(os.sched_getaffinity(0)))
    graphs = load_tg_dataset('movie','sssp')
    data_list = []
    dists_list = []
    dists_removed_list = []
    links_train_list = []
    links_val_list = []
    links_test_list = []
    for i, data in enumerate(graphs):
        print('culculate dist')
        start = time.time()
        dists_removed = precompute_dist_data(data, len(data.nodes),
                                         approximate=0)
        end = time.time()
        print('time: ',end-start)
