import numpy as np
import networkx as nx


def neighbor_degree(G, degree):
    """邻居度：邻居节点的度之和"""
    nodes = list(G.nodes())
    n_degree = np.zeros([1, len(nodes)])
    for node in range(len(nodes)):
        nd = 0
        neighbors = G.adj[node]
        for nei in neighbors:
            nd += degree[nei]
        n_degree[0, node] = nd
    return n_degree


def create_triangle_for_node(G: nx.Graph):
    triangle_list = []
    node_triangle_list = np.zeros([1, G.number_of_nodes()])
    larger_nei = {}
    for v in G.nodes:
        larger_nei[v] = {nei for nei in G.neighbors(v) if nei > v}

    for edge in G.edges:
        a = edge[0]
        b = edge[1]
        if a == b:
            continue
        com_nei = larger_nei[a] & larger_nei[b]

        for i in com_nei:  # 默认a是int
            node_triangle_list[0, a] += 1
            node_triangle_list[0, b] += 1
            node_triangle_list[0, i] += 1
            triangle_list.append([a, b, i])
    return node_triangle_list, triangle_list


def H_index(G: nx.Graph, degree):
    h_list = np.zeros(G.number_of_nodes())
    for node in G.nodes:
        neighbor_degree = []
        for neighbor in G.neighbors(node):
            neighbor_degree.append(degree[neighbor])
        min_degree = min(neighbor_degree)
        min_index = np.sum(np.array(neighbor_degree) == min_degree)
        h_list[int(node)] = min(min_index, min_degree)
    return h_list
