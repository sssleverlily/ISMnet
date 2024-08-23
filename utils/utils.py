import torch
import random
import numpy as np


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask


def random_planetoid_splits(data, args):
    train_split = args.train_rate
    val_split = args.val_rate
    node_num = data.num_nodes
    train_mask = np.zeros(node_num, dtype=np.bool)
    val_mask = np.zeros(node_num, dtype=np.bool)
    test_mask = np.zeros(node_num, dtype=np.bool)
    node_index = list(range(node_num))
    train_val_idx = random.sample(node_index, int((train_split + val_split) * node_num))
    val_idx = random.sample(train_val_idx, int(val_split * node_num))
    for index in train_val_idx:
        node_index.remove(index)
    for index in val_idx:
        train_val_idx.remove(index)
    train_mask[np.array(train_val_idx)] = True
    val_mask[np.array(val_idx)] = True
    test_mask[np.array(node_index)] = True
    data.train_mask = torch.tensor(train_mask)
    data.val_mask = torch.tensor(val_mask)
    data.test_mask = torch.tensor(test_mask)
    return data


def neighbor_degree(G):
    nodes = list(G.nodes())
    degree = dict(G.degree())
    n_degree = {}
    for node in nodes:
        nd = 0
        neighbors = G.adj[node]
        for nei in neighbors:
            nd += degree[nei]
        n_degree[node] = nd
    return n_degree
