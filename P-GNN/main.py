from sklearn.metrics import roc_auc_score
from tensorboardX import SummaryWriter
import sklearn.metrics

from args import *
from model import *
from utils import *
from dataset import *
import networkx as nx
import numpy as np
import sys
import torch
import torch.nn as nn
import time
import torch.functional as F

if __name__ == '__main__':
    start = time.time()
    if not os.path.isdir('results'):
        os.mkdir('results')
    # args
    args = make_args()
    print(args)
    np.random.seed(123)
    np.random.seed()
    writer_train = SummaryWriter(comment=args.task+'_'+args.model+'_'+args.comment+'_train')
    writer_val = SummaryWriter(comment=args.task+'_'+args.model+'_'+args.comment+'_val')
    writer_test = SummaryWriter(comment=args.task+'_'+args.model+'_'+args.comment+'_test')


    # set up gpu
    if args.gpu:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
        print('Using GPU {}'.format(os.environ['CUDA_VISIBLE_DEVICES']))
    else:
        print('Using CPU')
    #device = torch.device('cuda:'+str(args.cuda) if args.gpu else 'cpu')
    device = torch.device('cuda' if args.gpu else 'cpu')

    #for task in ['link', 'link_pair']:
    for task in ['link']:
        args.task = task
        if args.dataset=='All':
            if task == 'link':
                datasets_name = ['grid','communities','ppi']
            else:
                datasets_name = ['communities', 'email', 'protein']
        else:
            datasets_name = [args.dataset]
        for dataset_name in datasets_name:
            # if dataset_name in ['communities','grid']:
            #     args.cache = False
            # else:
            #     args.epoch_num = 401
            #     args.cache = True
            results = []
            for repeat in range(args.repeat_num):
                result_val = []
                result_test = []
                time1 = time.time()
                graphs, data_list = get_tg_dataset(args, dataset_name, use_cache=args.cache, remove_feature=args.rm_feature, type=args.type)
                time2 = time.time()
                print(dataset_name, 'load time',  time2-time1)

                num_features = data_list[0].x.shape[1]
                num_node_classes = None
                num_graph_classes = None
                if 'y' in data_list[0].__dict__ and data_list[0].y is not None:
                    num_node_classes = max([data.y.max().item() for data in data_list])+1
                if 'y_graph' in data_list[0].__dict__ and data_list[0].y_graph is not None:
                    num_graph_classes = max([data.y_graph.numpy()[0] for data in data_list])+1
                print('Dataset', dataset_name, 'Graph', len(data_list), 'Feature', num_features, 'Node Class', num_node_classes, 'Graph Class', num_graph_classes)
                nodes = [data.num_nodes for data in data_list]
                edges = [data.num_edges for data in data_list]
                print('Node: max{}, min{}, mean{}'.format(max(nodes), min(nodes), sum(nodes)/len(nodes)))
                print('Edge: max{}, min{}, mean{}'.format(max(edges), min(edges), sum(edges)/len(edges)))

                args.batch_size = min(args.batch_size, len(data_list))
                print('Anchor num {}, Batch size {}'.format(args.anchor_num, args.batch_size))

                # data
                for i,data in enumerate(data_list):
                    if args.model == 'PGNN':
                        preselect_anchor(data, layer_num=args.layer_num, anchor_num=args.anchor_num, device='cpu')
                        data = data.to(device)
                        data_list[i] = data

                # model
                input_dim = num_features
                output_dim = args.output_dim
                model = locals()[args.model](input_dim=input_dim, feature_dim=args.feature_dim,
                            hidden_dim=args.hidden_dim, output_dim=output_dim,
                            feature_pre=args.feature_pre, layer_num=args.layer_num, dropout=args.dropout).to(device)
                model = torch.nn.DataParallel(model)
                # loss
                optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
                if 'link' in args.task:
                    loss_func = nn.BCEWithLogitsLoss()
                    out_act = nn.Sigmoid()

                clip = True

                for epoch in range(args.epoch_num):
                    if epoch==200:
                        for param_group in optimizer.param_groups:
                            param_group['lr'] /= 10
                    model.train()
                    optimizer.zero_grad()
                    shuffle(data_list)
                    effective_len = len(data_list)//args.batch_size*len(data_list)
                    for id, data in enumerate(data_list[:effective_len]):
                        if args.permute and args.model == 'PGNN':
                            preselect_anchor(data, layer_num=args.layer_num, anchor_num=args.anchor_num, device=device)
                        data = data.to(device)

                        out = model(data)
                        #for var in data.edge_attr:
                        #    var.retain_grad()
                        # get_link_mask(data,resplit=False)  # resample negative links
                        edge_mask_train = np.concatenate((data.mask_link_positive_train, data.mask_link_negative_train), axis=-1)
                        nodes_first = torch.index_select(out, 0, torch.from_numpy(edge_mask_train[0,:]).long().to(device))
                        nodes_second = torch.index_select(out, 0, torch.from_numpy(edge_mask_train[1,:]).long().to(device))
                        pred = torch.sum(nodes_first * nodes_second, dim=-1)
                        label_positive = torch.ones([data.mask_link_positive_train.shape[1],], dtype=pred.dtype)
                        label_negative = torch.zeros([data.mask_link_negative_train.shape[1],], dtype=pred.dtype)
                        label = torch.cat((label_positive,label_negative)).to(device)
                        loss = loss_func(pred, label)

                        # update
                        loss.backward()

                        if clip:
                            nn.utils.clip_grad_norm_(model.parameters(), args.clip)

                        if id % args.batch_size == args.batch_size-1:
                            if args.batch_size>1:
                                # if this is slow, no need to do this normalization
                                for p in model.parameters():
                                    if p.grad is not None:
                                        p.grad /= args.batch_size
                            optimizer.step()
                            optimizer.zero_grad()


                    if epoch % args.epoch_log == 0:
                        # evaluate
                        model.eval()
                        loss_train = 0
                        loss_val = 0
                        loss_test = 0
                        correct_train = 0
                        all_train = 0
                        correct_val = 0
                        all_val = 0
                        correct_test = 0
                        all_test = 0
                        auc_train = 0
                        auc_val = 0
                        auc_test = 0
                        emb_norm_min = 0
                        emb_norm_max = 0
                        emb_norm_mean = 0
                        for id, data in enumerate(data_list):
                            out = model(data)
                            emb_norm_min += torch.norm(out.data, dim=1).min().cpu().numpy()
                            emb_norm_max += torch.norm(out.data, dim=1).max().cpu().numpy()
                            emb_norm_mean += torch.norm(out.data, dim=1).mean().cpu().numpy()

                            # train
                            # get_link_mask(data, resplit=False)  # resample negative links
                            edge_mask_train = np.concatenate((data.mask_link_positive_train, data.mask_link_negative_train), axis=-1)
                            nodes_first = torch.index_select(out, 0, torch.from_numpy(edge_mask_train[0, :]).long().to(device))
                            nodes_second = torch.index_select(out, 0, torch.from_numpy(edge_mask_train[1, :]).long().to(device))
                            pred = torch.sum(nodes_first * nodes_second, dim=-1)
                            label_positive = torch.ones([data.mask_link_positive_train.shape[1], ], dtype=pred.dtype)
                            label_negative = torch.zeros([data.mask_link_negative_train.shape[1], ], dtype=pred.dtype)
                            label = torch.cat((label_positive, label_negative)).to(device)
                            loss_train += loss_func(pred, label).cpu().data.numpy()
                            auc_train += roc_auc_score(label.flatten().cpu().numpy(), out_act(pred).flatten().data.cpu().numpy())
                            acc_train = culculate_acc(pred, label)
                            #if clip:
                            #clip = torch.max(out) > 2
                            train_labels = label.data.cpu().numpy()
                            train_pred = pred
                            # val
                            edge_mask_val = np.concatenate((data.mask_link_positive_val, data.mask_link_negative_val), axis=-1)
                            nodes_first = torch.index_select(out, 0, torch.from_numpy(edge_mask_val[0, :]).long().to(device))
                            nodes_second = torch.index_select(out, 0, torch.from_numpy(edge_mask_val[1, :]).long().to(device))
                            pred = torch.sum(nodes_first * nodes_second, dim=-1)
                            label_positive = torch.ones([data.mask_link_positive_val.shape[1], ], dtype=pred.dtype)
                            label_negative = torch.zeros([data.mask_link_negative_val.shape[1], ], dtype=pred.dtype)
                            label = torch.cat((label_positive, label_negative)).to(device)
                            loss_val += loss_func(pred, label).cpu().data.numpy()
                            auc_val += roc_auc_score(label.flatten().cpu().numpy(), out_act(pred).flatten().data.cpu().numpy())
                            acc_val = culculate_acc(pred, label)
                            val_labels = label.numpy()
                            val_pred = pred
                            # test
                            edge_mask_test = np.concatenate((data.mask_link_positive_test, data.mask_link_negative_test), axis=-1)
                            nodes_first = torch.index_select(out, 0, torch.from_numpy(edge_mask_test[0, :]).long().to(device))
                            nodes_second = torch.index_select(out, 0, torch.from_numpy(edge_mask_test[1, :]).long().to(device))
                            pred = torch.sum(nodes_first * nodes_second, dim=-1)
                            label_positive = torch.ones([data.mask_link_positive_test.shape[1], ], dtype=pred.dtype)
                            label_negative = torch.zeros([data.mask_link_negative_test.shape[1], ], dtype=pred.dtype)
                            label = torch.cat((label_positive, label_negative)).to(device)
                            loss_test += loss_func(pred, label).cpu().data.numpy()
                            auc_test += roc_auc_score(label.flatten().cpu().numpy(), out_act(pred).flatten().data.cpu().numpy())
                            acc_test = culculate_acc(pred, label)
                            test_labels = label.numpy()
                            test_pred = pred

                        loss_train /= id+1
                        loss_val /= id+1
                        loss_test /= id+1
                        emb_norm_min /= id+1
                        emb_norm_max /= id+1
                        emb_norm_mean /= id+1
                        auc_train /= id+1
                        auc_val /= id+1
                        auc_test /= id+1



                        print(repeat, epoch, 'Loss {:.4f}'.format(loss_train), 'Train AUC: {:.4f}'.format(auc_train),
                              'Val AUC: {:.4f}'.format(auc_val), 'Test AUC: {:.4f}'.format(auc_test),
                              'Train ACC: {:.4f}'.format(acc_train),
                              'Val ACC: {:.4f}'.format(acc_val), 'Test ACC: {:.4f}'.format(acc_test),
                              'Clip: {}'.format(clip), 'max: {:.4f}'.format(torch.max(out)),
                              )
                        writer_train.add_scalar('repeat_' + str(repeat) + '/auc_'+dataset_name, auc_train, epoch)
                        writer_train.add_scalar('repeat_' + str(repeat) + '/loss_'+dataset_name, loss_train, epoch)
                        writer_val.add_scalar('repeat_' + str(repeat) + '/auc_'+dataset_name, auc_val, epoch)
                        writer_train.add_scalar('repeat_' + str(repeat) + '/loss_'+dataset_name, loss_val, epoch)
                        writer_test.add_scalar('repeat_' + str(repeat) + '/auc_'+dataset_name, auc_test, epoch)
                        writer_test.add_scalar('repeat_' + str(repeat) + '/loss_'+dataset_name, loss_test, epoch)
                        writer_test.add_scalar('repeat_' + str(repeat) + '/emb_min_'+dataset_name, emb_norm_min, epoch)
                        writer_test.add_scalar('repeat_' + str(repeat) + '/emb_max_'+dataset_name, emb_norm_max, epoch)
                        writer_test.add_scalar('repeat_' + str(repeat) + '/emb_mean_'+dataset_name, emb_norm_mean, epoch)
                        result_val.append(auc_val)
                        result_test.append(auc_test)


                result_val = np.array(result_val)
                result_test = np.array(result_test)
                results.append(result_test[np.argmax(result_val)])
                G = graphs[0]
                adj_origin = nx.to_scipy_sparse_matrix(G).tocoo()
                adj = adj_origin.transpose()
                #adj = np.expand_dims(adj, axis=0)
                node_labels = data_list[0].x[:, 0]
                train_idx = np.concatenate((data.mask_link_positive_train,data.mask_link_negative_train), axis=-1)
                train_idx = np.swapaxes(train_idx,0, 1)
                test_idx = np.concatenate((data.mask_link_positive_val,
                                           data.mask_link_negative_val,
                                           data.mask_link_positive_test,
                                           data.mask_link_negative_test), axis=-1)
                test_idx = np.swapaxes(test_idx, 0, 1)
                if args.save == "embedding":
                    embeddings = model.get_embedding(data, explain=False)
                    nodes_first = torch.index_select(embeddings, 0, torch.from_numpy(edge_mask_train[0, :]).long().to(device))
                    nodes_second = torch.index_select(embeddings, 0, torch.from_numpy(edge_mask_train[1, :]).long().to(device))
                    train_pred = torch.cat((nodes_first, nodes_second), dim=-1)
                    nodes_first = torch.index_select(embeddings, 0, torch.from_numpy(edge_mask_val[0, :]).long().to(device))
                    nodes_second = torch.index_select(embeddings, 0, torch.from_numpy(edge_mask_val[1, :]).long().to(device))
                    val_pred = torch.cat((nodes_first, nodes_second), dim=-1)
                    nodes_first = torch.index_select(embeddings, 0, torch.from_numpy(edge_mask_test[0, :]).long().to(device))
                    nodes_second = torch.index_select(embeddings, 0, torch.from_numpy(edge_mask_test[1, :]).long().to(device))
                    test_pred = torch.cat((nodes_first, nodes_second), dim=-1)
                end = time.time()
                print('cost time: ', end-start)
                cg_data = {
                    "graph": G,
                    "adj": adj,
                    "feat": data.x.detach().numpy()[np.newaxis, :],
                    "node_labels": np.expand_dims(node_labels, axis=0),
                    "label_train": train_labels[np.newaxis, :],
                    "label_test": np.hstack((val_labels, test_labels))[np.newaxis, :],
                    "pred_train": train_pred.detach().numpy()[np.newaxis, :, np.newaxis] if args.save=="pred" else train_pred.detach().numpy()[np.newaxis, :],
                    "pred_test": np.hstack((val_pred.detach().numpy(), test_pred.detach().numpy()))[np.newaxis, :, np.newaxis] if args.save=="pred" \
                        else np.concatenate((val_pred.detach().numpy(), test_pred.detach().numpy()))[np.newaxis, :]   ,
                    "train_idx": torch.tensor(train_idx),
                    "test_idx": torch.tensor(test_idx),
                    "label_dic": G.labels,
                }
                save_checkpoint(model, optimizer, args, num_epochs=-1, cg_dict=cg_data)
                print("finish save checkpoint")
                """with open('../pred/pred_{}.txt'.format(args.dataset),'w') as save_pred:
                    for i in out:
                        for j in out:
                            if out_act(torch.sum(i * j, dim=-1)) > args.threshold:
                                save_pred.write("{}\t{}".format(i, j))"""
            results = np.array(results)
            results_mean = np.mean(results).round(6)
            results_std = np.std(results).round(6)
            print('-----------------Final-------------------')
            print(results_mean, results_std)
            with open('results/{}_{}_{}_layer{}_approximate{}.txt'.format(args.task,args.model,dataset_name,args.layer_num,args.approximate), 'w') as f:
                f.write('{}, {}\n'.format(results_mean, results_std))

    # export scalar data to JSON for external processing
    writer_train.export_scalars_to_json("./all_scalars.json")
    writer_train.close()
    writer_val.export_scalars_to_json("./all_scalars.json")
    writer_val.close()
    writer_test.export_scalars_to_json("./all_scalars.json")
    writer_test.close()