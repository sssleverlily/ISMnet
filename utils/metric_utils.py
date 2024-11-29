import itertools
import torch
import networkx as nx
import scipy.stats
from utils.dataset_utils import get_feature


def compare_tau(data, model, graph_name, hubOrder):
    sir_list = data.y.detach().cpu().numpy()
    result = model(data).detach().cpu().numpy()

    metrics = get_feature(graph_name=graph_name, hubOrder=hubOrder, metric=['dc', 'nd', 'ks', 'hIndex'])
    dc, nd, ks, hi = metrics[:, 0], metrics[:, 1], metrics[:, 2], metrics[:, 3]

    our_tau_list, _ = scipy.stats.kendalltau(result, sir_list)
    dc_tau_list, _ = scipy.stats.kendalltau(dc, sir_list)
    ks_tau_list, _ = scipy.stats.kendalltau(ks, sir_list)
    nd_tau_list, _ = scipy.stats.kendalltau(nd, sir_list)
    hi_tau_list, _ = scipy.stats.kendalltau(hi, sir_list)

    return our_tau_list, dc_tau_list, ks_tau_list, nd_tau_list, hi_tau_list


def new_rank_loss(y_true, y_pred):
    v_size = y_true.shape[0]
    temp = torch.arange(0, v_size).expand(v_size, -1)
    temp_0 = temp.reshape(-1)
    temp_1 = temp.t().reshape(-1)
    yt = y_true[temp_0] - y_true[temp_1]
    yt = (yt > 0) * 1.0 - (yt < 0) * 1.0
    yp = torch.tanh(y_pred[temp_0] - y_pred[temp_1])
    loss = 1 - torch.mean(yt * yp)

    return loss
