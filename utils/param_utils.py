import argparse
from models.ISMnet import ISMnet, CNN1
from models.benchmarks import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='hep', help='dataset name')
    parser.add_argument('--RPMAX', type=int, default=1, help='repeat times')
    parser.add_argument('--hubOrder', type=int, default=2, help='hub simplix dimension')
    parser.add_argument('--maxOrder', type=int, default=3, help='max simplix dimension')
    parser.add_argument('--beta1', type=int, default=1.0)
    parser.add_argument('--beta2', type=int, default=0.0)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--early_stopping', type=int, default=300)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.00001)
    parser.add_argument('--dprate', type=float, default=0.3)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--K', type=int, default=10)
    parser.add_argument('--train_rate', type=float, default=0.01)
    parser.add_argument('--val_rate', type=float, default=0.01)
    parser.add_argument('--hidden', type=int, default=32)
    parser.add_argument('--print_freq', type=int, default=20)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--is_gpu', type=bool, default=False)
    parser.add_argument('--net', type=str,
                        choices=['GCN', 'GAT', 'APPNP', 'ChebNet', 'JKNet', 'GPRGNN', 'BernNet', 'ISMnet',
                                 'CNN1'],
                        default='ISMnet',
                        )
    """
    The following arguments are used in the benchmarks!
    """
    parser.add_argument('--a', default=0.5, type=float)
    parser.add_argument('--b', default=1.0, type=float)
    parser.add_argument('--heads', default=8, type=int)
    parser.add_argument('--output_heads', default=1, type=int)
    parser.add_argument('--C', type=int)
    parser.add_argument('--Init', type=str, default='PPR')
    parser.add_argument('--Gamma', default=None)
    parser.add_argument('--ppnp', choices=['PPNP', 'GPR_prop'], default='GPR_prop')
    parser.add_argument('--Bern_lr', type=float, default=0.002, help='learning rate for BernNet propagation layer.')
    args = parser.parse_args()

    return args


def get_net(gnn_name):
    if gnn_name == 'BernNet':
        Net = BernNet
    elif gnn_name == 'ISMnet':
        Net = ISMnet
    elif gnn_name == 'CNN1':
        Net = CNN1
    return Net
