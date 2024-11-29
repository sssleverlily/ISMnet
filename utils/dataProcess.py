import copy
from utils.gen_HoHLaplacian import universal_HL_from_simplexlist
import networkx as nx
from utils.file_utils import *
import csv


def hIndex(neighbor_degree_list) -> int:
    """
    neighbor_degree_list: 单个节点的邻居度的list
    """
    neighbor_degree_list.sort()
    result = 0
    cir_num = len(neighbor_degree_list)
    for i in range(0, cir_num):
        if neighbor_degree_list[i] >= cir_num - i:
            result = cir_num - i
            break

    return result


def core_number(degree_list, nbrs):
    """
    Returns the core number for each vertex.
    degree_list:
    nbrs: list{v: v's neighbours}
    """
    nbrs = copy.deepcopy(nbrs)
    if isinstance(degree_list, list):
        n = len(degree_list)
    elif isinstance(degree_list, np.ndarray) or isinstance(degree_list, torch.Tensor):
        n = degree_list.shape[0]

    degrees = dict(zip(range(n), degree_list))
    # Sort nodes by degree.
    nodes = sorted(degrees, key=degrees.get)
    bin_boundaries = [0]
    curr_degree = 0
    for i, v in enumerate(nodes):
        if degrees[v] > curr_degree:
            bin_boundaries.extend([i] * (degrees[v] - curr_degree))
            curr_degree = degrees[v]
    node_pos = {v: pos for pos, v in enumerate(nodes)}
    core = degrees
    # nbrs = {v: list(nx.all_neighbors(G, v)) for v in G}
    for v in nodes:
        for u in nbrs[v]:
            if core[u] > core[v]:
                nbrs[u].remove(v)
                pos = node_pos[u]
                bin_start = bin_boundaries[core[u]]
                node_pos[u] = bin_start
                node_pos[nodes[bin_start]] = pos
                nodes[bin_start], nodes[pos] = nodes[pos], nodes[bin_start]
                bin_boundaries[core[u]] += 1
                core[u] -= 1
    return [core[v] for v in range(n)]


def graph_map(original_path, processed_path, map_path):
    """
    original_path: 原网络文件位置
    processed_path:处理后网络文件位置
    map_path:      映射关系文件位置
    使用范例：
    name = 'test_net'
    original_path  = os.path.join('..', 'data', name, name + '_original.txt')
    processed_path = os.path.join('..', 'data', name, name + '.txt')
    map_path       = os.path.join('..', 'data', name, 'node-index.csv')
    graph_map(original_path, processed_path, map_path)
    """

    G = nx.read_edgelist(original_path, create_using=nx.Graph(), nodetype=str)
    G.remove_edges_from(nx.selfloop_edges(G))
    G.remove_nodes_from(list(nx.isolates(G)))

    node_dict = {}  # 创造一个int和str的映射
    node_temp = 0
    for node in G.nodes:
        node_dict[node] = node_temp
        node_temp += 1

    # 将映射关系(节点-index)写入文件
    dataframe = pd.DataFrame({'simplex': list(node_dict.keys()), 'index': list(node_dict.values())})
    dataframe.to_csv(map_path, mode="w", index=False)

    # 将映射后的网络写入文件
    with open(processed_path, 'w') as f:
        for edge in G.edges:
            f.write("{} {}\n".format(node_dict[edge[0]], node_dict[edge[1]]))


def gen_simplex_list(name, maxOrder):
    """
    对于单复形网络不用执行该函数
    对于一般的pairwise网络需要先执行该函数，得到所有单纯形的list（每行一个单纯形，行号就是单纯形的index）
    """
    for k_ in range(maxOrder + 1):
        if os.path.exists(get_simplex_list_path(name, k_)):
            os.remove(get_simplex_list_path(name, k_))

    G = nx.read_edgelist(get_net_path(name), create_using=nx.Graph(), nodetype=int)
    n_ = np.zeros(maxOrder + 1, dtype=int)
    n_[0] = G.number_of_nodes()

    itClique = nx.enumerate_all_cliques(G)
    nextClique = next(itClique)
    while len(nextClique) <= 1:
        try:
            nextClique = next(itClique)
        except StopIteration:
            break

    while len(nextClique) <= maxOrder + 1:
        try:
            order = len(nextClique) - 1
            n_[order] += 1
            with open(get_simplex_list_path(name, order), 'a+') as f:
                writer = csv.writer(f)
                writer.writerow(nextClique)

            nextClique = next(itClique)
            nextClique.sort()
            print(nextClique)
        except StopIteration:
            print("转运层的最大阶数小于枢纽层阶数！")
            break

    # 记录各阶单纯形的数量
    with open(get_statistic_path(name), "w") as f:
        statistics = {'n_simplex': n_.tolist()}
        yaml.dump(statistics, f)


def calucate_HND_HIndex_HCoreness(name, hubOrder, maxOrder):
    HL = torch.load(get_hl_path(name, hubOrder))

    Hdegree_path = get_Hdegree_path(name, hubOrder)
    Hdegree = pd.read_csv(Hdegree_path)
    n_hubOrder = Hdegree.shape[0]

    HND, HhIndex, Hcoreness = {}, {}, {}
    nbrs = {}

    for k_ in range(0, maxOrder + 1):
        print("calucate index in order={} (hubOrder={})".format(k_, hubOrder))

        if k_ == hubOrder: continue
        HND['{},{}-nd'.format(k_, hubOrder)] = torch.zeros(n_hubOrder, dtype=torch.int)
        HhIndex['{},{}-HhIndex'.format(k_, hubOrder)] = torch.zeros(n_hubOrder, dtype=torch.int)

        for node in range(0, n_hubOrder):
            nbrs[node] = torch.cat((HL[k_][node, 0:node].coo()[1],
                                    HL[k_][node, node + 1: n_hubOrder + 1].coo()[1] + node + 1), dim=0).tolist()

        _hd = Hdegree['{},{}-d'.format(k_, hubOrder)].values

        Hcoreness['{},{}-HCore'.format(k_, hubOrder)] = core_number(_hd, nbrs)
        for node in range(0, n_hubOrder):
            HND['{},{}-nd'.format(k_, hubOrder)][node] = sum(_hd[nbrs[node]])
            HhIndex['{},{}-HhIndex'.format(k_, hubOrder)][node] = hIndex(_hd[nbrs[node]])

    # 把HND, Hcoreness, HhIndex写入文件
    df = pd.DataFrame(HND)
    df.to_csv(get_HND_path(name, hubOrder), mode="w", index=False)

    df = pd.DataFrame(Hcoreness)
    df.to_csv(get_HCoreness_path(name, hubOrder), mode="w", index=False)

    df = pd.DataFrame(HhIndex)
    df.to_csv(get_HhIndex_path(name, hubOrder), mode="w", index=False)

    print("Finish calucate fatures  ({}, hubOrder={}, maxOrder={})".format(name, hubOrder, maxOrder))


def pre_process(name, hubOrder, maxOrder):
    """
    name: 网络的名称
    hubOrder: 枢纽层的阶数
    maxOrder: 考虑的最大单纯形的阶数
    """

    # 新建目录
    gen_dir(name, maxOrder)

    base_path = get_base_path(name, hubOrder)
    """
    将原始网络过一层映射
    """
    net_path = get_net_path(name)  # 映射后的网络
    if not os.path.exists(net_path):
        orgNet = os.path.join('..', 'data', name, name + '_original.txt')  # 原始网络
        map_path = os.path.join('..', 'data', name, 'node-index.csv')
        graph_map(original_path=orgNet, processed_path=net_path, map_path=map_path)

    if not os.path.exists(get_statistic_path(name)):
        # 这个函数会生产两个文件：simplex_list和统计信息
        gen_simplex_list(name, maxOrder)
        print("Finish gen simplex list of {}!".format(name))

    HL, Hdegree = universal_HL_from_simplexlist(name, hubOrder, maxOrder)

    # 把HL存下来
    hl_path = get_hl_path(name, hubOrder)  # 存储hl的路径
    torch.save(HL, hl_path)

    dataframe = pd.DataFrame(Hdegree)
    dataframe.to_csv(get_Hdegree_path(name, hubOrder), mode="w", index=False)

    print("Finish pre-process ({}, hubOrder={}, maxOrder={})".format(name, hubOrder, maxOrder))


if __name__ == '__main__':
    dataset = 'figeys'
    maxOrder = 3  # 最大的阶数

    for hubOrder in [0, 1, 2]:
        print("hubOrder={}".format(hubOrder))
        pre_process(dataset, hubOrder, maxOrder)
        calucate_HND_HIndex_HCoreness(dataset, hubOrder, maxOrder)
