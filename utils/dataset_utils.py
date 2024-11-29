import os
import pandas as pd
from torch_geometric.data import InMemoryDataset, download_url, Data
from utils.indicators import *
import torch
import networkx as nx
from utils.file_utils import *
from utils.utils import *
import community as community_louvain


def DataLoader_ISM(name, args):
    if name not in ['facebook', 'figeys', 'GrQ', 'ham', 'hep', 'powergrid', 'vidal', 'LastFM', 'Sex', 'test_net']:
        raise ValueError(f'dataset {name} not supported in dataloader')

    HL = torch.load(get_hl_path(name, args.hubOrder))

    y = get_SIR(name, args.hubOrder, args.beta1, args.beta2)
    if args.hubOrder < 2:
        Hdegree = torch.from_numpy(pd.read_csv(get_Hdegree_path(name, args.hubOrder)).values)
        HND = torch.from_numpy(pd.read_csv(get_HND_path(name, args.hubOrder)).values)
        HhIndex = torch.from_numpy(pd.read_csv(get_HhIndex_path(name, args.hubOrder)).values)
        HCore = torch.from_numpy(pd.read_csv(get_HCoreness_path(name, args.hubOrder)).values)
    else:
        Hdegree, HND, HhIndex, HCore = generate_feature_2_simplex(name, 2)
        Hdegree, HND, HhIndex, HCore = Hdegree.view(-1, 1), HND.view(-1, 1), HhIndex.view(-1, 1), HCore.view(-1, 1)

    data = Data(x=torch.cat((Hdegree, HND, HhIndex, HCore), dim=1).float(), HL=HL, y=y.view(-1, 1))
    print(data)

    print("Finish load data!")
    return data


def get_feature(graph_name, hubOrder, metric):
    "若metric是字符串"
    if isinstance(metric, str):
        metric = [metric]

    rel = torch.tensor([])
    "metric是字符串list"
    for mm in metric:
        if mm in ['degree', 'dc']:
            t = torch.from_numpy(pd.read_csv(get_Hdegree_path(graph_name, hubOrder)).values[:, 0])
        elif mm in ['ND', 'neighbor_degree', 'nd']:
            t = torch.from_numpy(pd.read_csv(get_HND_path(graph_name, hubOrder)).values[:, 0])
        elif mm in ['hIndex']:
            t = torch.from_numpy(pd.read_csv(get_HhIndex_path(graph_name, hubOrder)).values[:, 0])
        elif mm in ['coreness', 'ks']:
            t = torch.from_numpy(pd.read_csv(get_HCoreness_path(graph_name, hubOrder)).values[:, 0])
        else:
            print("Graph {} has not stored {} yet!".format(graph_name, mm))

        rel = torch.concat((rel, t.view(-1, 1)), dim=1)

    return rel


def generate_feature_2_simplex(name, order):
    degree_list = get_feature(name, 0, 'degree')
    nd_list = get_feature(name, 0, 'nd')
    hi_list = get_feature(name, 0, 'hIndex')
    ks_list = get_feature(name, 0, 'ks')
    matrix = pd.read_csv(os.path.join(get_base_path(name, order), 'simplex-index.csv'), header=None)
    degree_result = torch.zeros(matrix.shape[0])
    nd_result = torch.zeros(matrix.shape[0])
    hi_result = torch.zeros(matrix.shape[0])
    ks_result = torch.zeros(matrix.shape[0])
    for x in range(matrix.shape[0]):
        for y in range(matrix.shape[1]):
            degree_result[x] = degree_result[x] + degree_list[matrix.T[x][y]] / (order + 1)
            nd_result[x] = nd_result[x] + nd_list[matrix.T[x][y]] / (order + 1)
            hi_result[x] = hi_result[x] + hi_list[matrix.T[x][y]] / (order + 1)
            ks_result[x] = ks_result[x] + ks_list[matrix.T[x][y]] / (order + 1)
    return degree_result, nd_result, hi_result, ks_result
