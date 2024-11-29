import torch
import numpy as np
import networkx as nx
from torch_sparse import SparseTensor, fill_diag, matmul, mul
import math
import pandas as pd
from itertools import combinations
import os
from utils.file_utils import *


def universal_HL_from_simplexlist(name, hubOrder=0, maxOrder=3):
    if os.path.exists(get_statistic_path(name)):
        with open(get_statistic_path(name), "r") as f:
            n_ = yaml.safe_load(f)['n_simplex']

    ## 映射字典,无需记录maxOrder阶单纯形的映射
    # simplex_dict = {k_: {} for k_ in range(0, maxOrder)}
    # 记得存simplex_list的时候要先排序
    if hubOrder >= 1:
        hub_simplex_dict = dict(zip([tuple(ss) for ss in get_simplex_list(name, hubOrder)], range(n_[hubOrder])))
    elif hubOrder == 0:
        hub_simplex_dict = dict(zip([(i,) for i in range(n_[0])], range(n_[0])))

    # _degree[hub_order]:枢纽单纯形度
    _degree, _D, L = {}, {}, {}
    for i in range(0, maxOrder + 1):
        if i == hubOrder: continue
        _degree['{},{}-d'.format(i, hubOrder)] = torch.zeros(n_[hubOrder], dtype=torch.int)

    for k_ in range(0, hubOrder):
        print("gen HL in order={}...".format(k_))
        if k_ == 0:
            t_simplex_dict = dict(zip([(i,) for i in range(n_[0])], range(n_[0])))
        else:
            t_simplex_dict = dict(zip([tuple(ss) for ss in get_simplex_list(name, k_)], range(n_[k_])))

        _D = torch.zeros(n_[k_])
        H = SparseTensor(row=torch.LongTensor([]), col=torch.LongTensor([]), value=torch.tensor([]),
                         sparse_sizes=(n_[hubOrder], n_[k_]))

        ones_ = torch.ones(math.comb(hubOrder + 1, k_ + 1))
        for sidx, simplex in enumerate(hub_simplex_dict.keys()):
            print("gen HL in dealing with {}".format(simplex))
            for sub in combinations(simplex, k_ + 1):
                _D[t_simplex_dict[sub]] += 1
            cols = torch.LongTensor([t_simplex_dict[sub] for sub in combinations(simplex, k_ + 1)])
            H = H + SparseTensor(row=ones_.long() * sidx, col=cols, value=ones_, sparse_sizes=(n_[hubOrder], n_[k_]))

        _D[_D == 0] = 1
        _D = 1 / _D
        sub_degree = SparseTensor(row=torch.LongTensor(list(range(n_[k_]))),
                                  col=torch.LongTensor(list(range(n_[k_]))),
                                  value=_D,  # 不知道为什么设置了value没有用
                                  sparse_sizes=(n_[k_], n_[k_]))

        A = H.matmul(H.t())
        for i in range(0, n_[hubOrder]):
            _degree['{},{}-d'.format(k_, hubOrder)][i] = sum(A[i, :].coo()[-1]) - hubOrder - 1

        L[k_] = H.matmul(sub_degree).matmul(H.t())
        _D = torch.ones(n_[hubOrder]) * (1 / math.comb(hubOrder + 1, k_ + 1))
        L[k_] = mul(L[k_], _D.view(1, -1))
        L[k_] = mul(L[k_], _D.view(-1, 1))

    for k_ in range(hubOrder + 1, maxOrder + 1):
        print("gen HL in order={}...".format(k_))
        t_simplex_list = get_simplex_list(name, k_)

        _D = torch.zeros(n_[hubOrder])
        H = SparseTensor(row=torch.LongTensor([]), col=torch.LongTensor([]), value=torch.tensor([]),
                         sparse_sizes=(n_[hubOrder], n_[k_]))

        ones_ = torch.ones(math.comb(k_ + 1, hubOrder + 1))
        for sidx, simplex in enumerate(t_simplex_list):
            print("gen HL in dealing with {}".format(simplex))
            for sub in combinations(simplex, hubOrder + 1):
                _D[hub_simplex_dict[sub]] += 1
            rows = torch.LongTensor([hub_simplex_dict[sub] for sub in combinations(simplex, hubOrder + 1)])
            H = H + SparseTensor(row=rows, col=ones_.long() * sidx, value=ones_, sparse_sizes=(n_[hubOrder], n_[k_]))

        A = H.matmul(H.t())
        for i in range(0, n_[hubOrder]):
            _degree['{},{}-d'.format(k_, hubOrder)][i] = sum(A[i, :].coo()[-1])
        _degree['{},{}-d'.format(k_, hubOrder)] = _degree['{},{}-d'.format(k_, hubOrder)] - _D.int()

        _D[_D == 0] = 1
        _D = _D.pow_(-0.5)
        L[k_] = mul(A, _D.view(1, -1))
        L[k_] = mul(L[k_], _D.view(-1, 1) / (k_ + 1))

    return L, _degree
