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

def gen_feature(name, G: nx.Graph(), origin_str_graph=None, node_dic=None):
    """ ----------
    G : 节点类型int
    origin_str_graph: 节点类型str
    node_dic : origin_str_graph 和 G 节点间的映射关系
    ------- """

    partition = community_louvain.best_partition(G)
    pt_list = [partition[i] for i in range(G.number_of_nodes())]
    one_hot = torch.diag(torch.ones(max(pt_list) + 1))

    path = os.path.join('..', 'data', name, name + '_struc.pt')
    struc = torch.load(path)

    feature_num = 5 + max(pt_list) + 1 + struc[list(origin_str_graph.nodes)[0]].size

    feat = torch.zeros((G.number_of_nodes(), feature_num), dtype=torch.float)
    cn = nx.core_number(G)
    dc = nx.degree(G)
    dc_list = [dc[i] for i in range(G.number_of_nodes())]
    neighbor_degree_list = neighbor_degree(G, dc_list)
    h_list = H_index(G, dc_list)
    node_triangle = create_triangle_for_node(G)[0]

    for node in G.nodes:
        feat[int(node)][0] = cn[node]
        feat[int(node)][1] = dc[node]
        feat[int(node)][2] = neighbor_degree_list[0, node]
        feat[int(node)][3] = node_triangle[0, node]
        feat[int(node)][4] = h_list[node]
        feat[int(node), 5: 5 + max(pt_list) + 1] = one_hot[pt_list[int(node)]]
    for node in range(origin_str_graph.number_of_nodes()):
        feat[node_dic[list(origin_str_graph.nodes)[node]], (5 + max(pt_list) + 1): feature_num] = struc[
            list(origin_str_graph.nodes)[node]].size
    return feat


def get_feature(graph_name, hubOrder, metric):
    "若metric是字符串"
    if isinstance(metric, str):
        metric = [metric]

    rel = torch.tensor([])
    "metric是字符串list"
    for mm in metric:
        if mm in ['degree', 'dc']:
            t = torch.from_numpy(pd.read_csv(get_Hdegree_path(graph_name, hubOrder)).values[:, 0])
        elif mm in ['ND', 'neighbor_degree','nd']:
            t = torch.from_numpy(pd.read_csv(get_HND_path(graph_name, hubOrder)).values[:, 0])
        elif mm in ['hIndex']:
            t = torch.from_numpy(pd.read_csv(get_HhIndex_path(graph_name, hubOrder)).values[:, 0])
        elif mm in ['coreness', 'ks']:
            t = torch.from_numpy(pd.read_csv(get_HCoreness_path(graph_name, hubOrder)).values[:, 0])
        else:
            print("Graph {} has not stored {} yet!".format(graph_name, mm))

        rel = torch.concat((rel, t.view(-1, 1)), dim=1)

    return rel


def generate_graph(path):
    G = nx.read_edgelist(path, create_using=nx.Graph(), nodetype=str)
    # graph = nx.read_edgelist(path, create_using=nx.Graph(), nodetype=int)
    node_dic = {}  # 创造一个int和str的映射
    node_temp = 0
    for node in G.nodes:
        node_dic[node] = node_temp
        node_temp += 1
    graph = nx.Graph()
    for edge in G.edges:
        graph.add_edge(node_dic[edge[0]], node_dic[edge[1]])
    return graph, node_dic, G


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