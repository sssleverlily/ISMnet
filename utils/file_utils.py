import os
import pandas as pd
import yaml
import numpy as np
import torch


def gen_dir(name, maxOrder=3):
    """
    生成呈现运行必要的目录
    """
    for k_ in range(0, maxOrder + 1):
        base_path = get_base_path(name, k_)
        sir_path = os.path.join(base_path, 'SIR_data')
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        if not os.path.exists(sir_path):
            os.makedirs(sir_path)


def get_base_path(name, hubOrder):
    return os.path.join('..', 'data', name, '{}-simplex'.format(hubOrder))


def get_net_path(name):
    return os.path.join('..', 'data', name, name + '.txt')


def get_hl_path(name, hubOrder):
    return os.path.join(get_base_path(name, hubOrder), 'HL_{}.pt'.format(name))


# 里面记录了各阶单纯形的数量
def get_statistic_path(name):
    return os.path.join('..', 'data', name, name + '_statistics.yaml')


# simplex_dict的存储路径
# 读文件时候要注意加上header=None，否则会把第一行看作数header
# pd.read_csv(get_simplex_dict_path(name, Order=2), header=None).values
def get_simplex_list_path(name, Order):
    return os.path.join('..', 'data', name, '{}-simplex'.format(Order), 'simplex-index.csv')


def get_simplex_list(name, Order):
    if Order <= 0:
        with open(get_statistic_path(name), "r") as f:
            n0 = yaml.safe_load(f)['n_simplex'][0]
        return list(range(n0))
    else:
        return pd.read_csv(get_simplex_list_path(name, Order), header=None).values


def get_Hdegree_path(name, hubOrder):
    return os.path.join(get_base_path(name, hubOrder), 'Hdegree.csv')


def get_HND_path(name, hubOrder):
    return os.path.join(get_base_path(name, hubOrder), 'HND.csv')


def get_HhIndex_path(name, hubOrder):
    return os.path.join(get_base_path(name, hubOrder), 'HhIndex.csv')


def get_HCoreness_path(name, hubOrder):
    return os.path.join(get_base_path(name, hubOrder), 'HCoreness.csv')


# 生成SIR标签的路径
def get_SIR_path(name, hubOrder, beta1, beta2):
    return os.path.join(get_base_path(name, hubOrder), 'SIR_data', '{}_{}_{}.csv'.format(name, beta1, beta2))


# SIR_path中每一行代表一万次的均值，再对这些均值取平均
def get_SIR(name, hubOrder, beta1, beta2):
    sir = pd.read_csv(get_SIR_path(name, hubOrder, beta1, beta2), header=None).values
    if len(sir.shape) > 1 and len(sir[1]) > 1:
        # 若是原始数据（未取均值，每行是一条记录）
        return torch.from_numpy(np.mean(sir, axis=0)).view(-1, 1)
    else:
        return torch.from_numpy(sir)
