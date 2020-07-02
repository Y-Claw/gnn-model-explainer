""" explainer_main.py

     Main user interface for the explainer module.
"""
import argparse
import os

import sklearn.metrics as metrics

from tensorboardX import SummaryWriter

import pickle
import shutil
import torch

import models
import utils.io_utils as io_utils
import utils.parser_utils as parser_utils
from explainer import explain
import sys


def arg_parse():
    parser = argparse.ArgumentParser(description="GNN Explainer arguments.")
    io_parser = parser.add_mutually_exclusive_group(required=False)
    io_parser.add_argument("--dataset", dest="dataset", help="Input dataset.")
    benchmark_parser = io_parser.add_argument_group()
    benchmark_parser.add_argument(
        "--bmname", dest="bmname", help="Name of the benchmark dataset"
    )
    io_parser.add_argument("--pkl", dest="pkl_fname", help="Name of the pkl data file")

    parser_utils.parse_optimizer(parser)

    parser.add_argument("--clean-log", action="store_true", help="If true, cleans the specified log directory before running.")
    parser.add_argument("--logdir", dest="logdir", help="Tensorboard log directory")
    parser.add_argument("--ckptdir", dest="ckptdir", help="Model checkpoint directory")
    parser.add_argument("--cuda", dest="cuda", help="CUDA.")
    parser.add_argument(
        "--gpu",
        dest="gpu",
        action="store_const",
        const=True,
        default=False,
        help="whether to use GPU.",
    )
    parser.add_argument(
        "--epochs", dest="num_epochs", type=int, help="Number of epochs to train."
    )
    parser.add_argument(
        "--hidden-dim", dest="hidden_dim", type=int, help="Hidden dimension"
    )
    parser.add_argument(
        "--output-dim", dest="output_dim", type=int, help="Output dimension"
    )
    parser.add_argument(
        "--num-gc-layers",
        dest="num_gc_layers",
        type=int,
        help="Number of graph convolution layers before each pooling",
    )
    parser.add_argument(
        "--bn",
        dest="bn",
        action="store_const",
        const=True,
        default=False,
        help="Whether batch normalization is used",
    )
    parser.add_argument("--dropout", dest="dropout", type=float, help="Dropout rate.")
    parser.add_argument(
        "--nobias",
        dest="bias",
        action="store_const",
        const=False,
        default=True,
        help="Whether to add bias. Default to True.",
    )
    parser.add_argument(
        "--no-writer",
        dest="writer",
        action="store_const",
        const=False,
        default=True,
        help="Whether to add bias. Default to True.",
    )
    # Explainer
    parser.add_argument("--mask-act", dest="mask_act", type=str, help="sigmoid, ReLU.")
    parser.add_argument(
        "--mask-bias",
        dest="mask_bias",
        action="store_const",
        const=True,
        default=False,
        help="Whether to add bias. Default to True.",
    )
    parser.add_argument(
        "--explain-node", dest="explain_node", type=int, help="Node to explain."
    )
    parser.add_argument(
        "--graph-idx", dest="graph_idx", type=int, help="Graph to explain."
    )
    parser.add_argument(
        "--graph-mode",
        dest="graph_mode",
        action="store_const",
        const=True,
        default=False,
        help="whether to run Explainer on Graph Classification task.",
    )
    parser.add_argument(
        "--multigraph-class",
        dest="multigraph_class",
        type=int,
        help="whether to run Explainer on multiple Graphs from the Classification task for examples in the same class.",
    )
    parser.add_argument(
        "--multinode-class",
        dest="multinode_class",
        type=int,
        help="whether to run Explainer on multiple nodes from the Classification task for examples in the same class.",
    )
    parser.add_argument(
        "--align-steps",
        dest="align_steps",
        type=int,
        help="Number of iterations to find P, the alignment matrix.",
    )

    parser.add_argument(
        "--method", dest="method", type=str, help="Method. Possible values: base, att."
    )
    parser.add_argument(
        "--name-suffix", dest="name_suffix", help="suffix added to the output filename"
    )
    parser.add_argument(
        "--explainer-suffix",
        dest="explainer_suffix",
        help="suffix added to the explainer log",
    )

    parser.add_argument('--link_prediction', dest='link_prediction',
                        help='whether do link prediction task')
    parser.add_argument('--directed_graph', dest='directed_graph',
                        help='whether graph is directed')
    parser.add_argument('--single_edge_label', dest='single_edge_label',
                        help='whether there is only one type of edges in the graph')
    parser.add_argument('--multi_class', dest='multi_class',
                        help='whether multi class classification for link prediction')
    parser.add_argument('--multi_label', dest='multi_label',
                        help='whether multi label classification for link prediction')
    parser.add_argument('--n_hops', dest='n_hops',
                        help='n hops neighboors')
    parser.add_argument('--edge_threshold', dest='edge_threshold',
                        help='the edge threshold for filtering during explanation')
    parser.add_argument('--feat_threshold', dest='feat_threshold',
                        help='the feature threshold for filtering during explanation')
    parser.add_argument('--max_edges_num', dest='max_edges_num',
                        help='the max number of edges of the explanation results')

    # TODO: Check argument usage
    parser.set_defaults(
        # mainly change the following arguments:
        gpu=False,
        logdir="log",
        ckptdir="ckpt",
        dataset="USAir",
        directed_graph=True,
        link_prediction=True,
        single_edge_label=True,
        multi_class=False,
        multi_label=False,
        n_hops=2,
        lr=0.1,
        num_epochs=1000,
        edge_threshold=0.5,
        feat_threshold=0.5,
        max_edges_num=15,
        writer=False,

        # no change is ok for the following arguments:
        opt="adam",
        opt_scheduler="none",
        cuda="0",
        clip=2.0,
        batch_size=20,
        hidden_dim=20,
        output_dim=20,
        num_gc_layers=3,
        dropout=0.0,
        method="base",
        name_suffix="",
        explainer_suffix="",
        align_steps=1000,
        explain_node=None,
        graph_idx=-1,
        mask_act="sigmoid",
        multigraph_class=-1,
        multinode_class=-1,
    )
    return parser.parse_args()


def main():
    # Load a configuration
    prog_args = arg_parse()

    if prog_args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = prog_args.cuda
        print("CUDA", prog_args.cuda)
    else:
        print("Using CPU")

    # Configure the logging directory 
    if prog_args.writer:
        path = os.path.join(prog_args.logdir, io_utils.gen_explainer_prefix(prog_args))
        if os.path.isdir(path) and prog_args.clean_log:
           print('Removing existing log dir: ', path)
           if not input("Are you sure you want to remove this directory? (y/n): ").lower().strip()[:1] == "y": sys.exit(1)
           shutil.rmtree(path)
        writer = SummaryWriter(path)
    else:
        writer = None

    # Load a model checkpoint
    ckpt = io_utils.load_ckpt(prog_args)
    cg_dict = ckpt["cg"]  # get computation graph
    input_dim = cg_dict["feat"].shape[2] 
    num_classes = cg_dict["pred_train"].shape[2]
    print("Loaded model from {}".format(prog_args.ckptdir))
    print("input dim: ", input_dim, "; num classes: ", num_classes)

    # build model
    print("Method: ", prog_args.method)
    if prog_args.link_prediction is True:
        model = models.GcnEncoderNode(
            input_dim=input_dim,
            hidden_dim=prog_args.hidden_dim,
            embedding_dim=prog_args.output_dim,
            label_dim=num_classes,
            num_layers=prog_args.num_gc_layers,
            bn=prog_args.bn,
            args=prog_args,
        )
    if prog_args.gpu:
        model = model.cuda()
    # load state_dict (obtained by model.state_dict() when saving checkpoint)
    model.load_state_dict(ckpt["model_state"]) 

    # Create explainer
    explainer = explain.Explainer(
        model=model,
        graph=cg_dict["graph"],
        adj=cg_dict["adj"],
        feat=cg_dict["feat"],
        node_labels=cg_dict["node_labels"],
        train_labels=cg_dict["label_train"],
        test_labels=cg_dict["label_test"],
        pred_train=cg_dict["pred_train"],
        pred_test=cg_dict["pred_test"],
        train_idx=cg_dict["train_idx"],
        test_idx=cg_dict["test_idx"],
        args=prog_args,
        writer=writer,
        print_training=True,
        graph_mode=False,
        graph_idx=prog_args.graph_idx,
    )
    # adj[u][v] = 1 means that, there's an edge from node v to node u. (directed graph)

    if prog_args.link_prediction is True:
        print("begin explaining a set of links one by one...")
        src_explain_res, dst_explain_res, src_denoise_res, dst_denoise_res = explainer.explain_a_set_of_links(
            prog_args
        )
        print("finish explaining all links.")


if __name__ == "__main__":
    main()

